import glob
import json
import os

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, create_map, lit, min, rank, round, concat
from pyspark.sql.window import Window
from itertools import chain

import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


def main():
    """The main function to extract, transform and send report."""
    config = load_config()
    spark = create_spark_session()

    report = extract_data(spark, config['data']['source'])
    report = map_vendor_payment_type(report)
    report = calculate_passenger_count_per_vendor_payment(report)
    report = calculate_payment_rate(report)
    report = add_next_payment_rate(report)
    report = calculate_max_payment_rate(report)
    report = calculate_percents_to_next_rate(report)
    report = select_fields(report)

    report_file = save_report(config['data']['destination'], report)
    send_report(config['mail'], report_file)


def load_config():
    """
    Read config
    Returns:
        map of config
    """
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.getenv('CONFIG_PATH') or os.path.join(location, 'config.json'), 'r') as f:
        return json.load(f)


def create_spark_session():
    """
    Creates spark session
    Returns:
        SparkSession object
    """
    config = SparkConf().setAppName('Driver').setMaster('local[*]')
    return SparkSession.builder.config(conf=config).getOrCreate()


def extract_data(spark, file):
    """
    Extract data from the csv file
    Arguments:
        spark: SparkSession
        file: File path to csv file
    Returns:
        Spark DataFrame
    """
    return spark.read \
        .option('header', True) \
        .option('inferSchema', True) \
        .csv(file, header=True)


def map_vendor_payment_type(report):
    """
    Map integer values to string for Vendor and Payment Type
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    vendors_dict = {1: 'Creative Mobile Technologies', 2: 'VeriFone Inc.'}
    vendors_mapping_expr = create_map([lit(x) for x in chain(*vendors_dict.items())])

    payment_types_dict = {1: 'Credit card', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided trip'}
    payment_types_mapping_expr = create_map([lit(x) for x in chain(*payment_types_dict.items())])

    return report \
        .withColumn('Vendor', vendors_mapping_expr[col('VendorID')]) \
        .withColumn('Payment Type', payment_types_mapping_expr[col('payment_type')])


def calculate_passenger_count_per_vendor_payment(report):
    """
    Calculates passenger count per vendor payment as PCPVP column
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    pass_count_per_vendor_payment = report \
        .groupBy('VendorID', 'payment_type').sum('passenger_count') \
        .withColumnRenamed("sum(passenger_count)", 'PCPVP')

    return report.join(pass_count_per_vendor_payment, ['VendorID', 'payment_type'])


def calculate_payment_rate(report):
    """
    Calculates pyament rate based on total amount and passenger count
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    return report.withColumn('Payment Rate', col('total_amount') / col('PCPVP'))


def add_next_payment_rate(report):
    """
    Adds column with next payment rate, for that first adds rank and search next rank payment rate value
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    window_spec_rank = Window.partitionBy('VendorID', 'payment_type').orderBy('Payment Rate')
    window_spec_next_rank = Window.partitionBy('VendorID', 'payment_type').orderBy('payment_rate_rank') \
        .rangeBetween(1, Window.unboundedFollowing)

    return report \
        .withColumn('payment_rate_rank', rank().over(window_spec_rank)) \
        .withColumn('Next Payment Rate', min('Payment Rate').over(window_spec_next_rank))


def calculate_max_payment_rate(report):
    """
    Calculates max payment rate per vendor
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    max_payment_rate_per_vendor = report \
        .groupBy('VendorID').max('Payment Rate') \
        .withColumnRenamed("max(Payment Rate)", 'Max Payment Rate')

    return report.join(max_payment_rate_per_vendor, 'VendorID')


def calculate_percents_to_next_rate(report):
    """
    Calculates percents to next rate and round value to two decimal
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    return report.withColumn(
        'Percents to next rate',
        concat(
            round(100 - col('Payment Rate') * 100 / col('Next Payment Rate'), 2),
            lit('%')
        )
    )


def select_fields(report):
    """
    Selects only required columns
    Arguments:
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    return report.select('Vendor', 'Payment Type', 'Payment Rate', 'Next Payment Rate', 'Max Payment Rate',
                         'Percents to next rate')


def save_report(destination, report):
    """
    Saves provided DataFrame to the destination as single file
    Arguments:
        destination: File path
        report: Spark DataFrame
    Returns:
        Spark DataFrame
    """
    report.coalesce(1).write.mode('overwrite').option('header', 'true').csv(destination)

    list_of_files = glob.glob(f'{destination}/*.csv')
    return max(list_of_files, key=os.path.getctime)


def send_report(config, file):
    """
    Sends report to the email from config
    Arguments:
        config: Config dictionary with email parameters
        file: Report CSV file to send
    Returns:
        None
    """
    msg = MIMEMultipart()
    msg['Subject'] = config['subject']
    msg['From'] = config['user']
    msg['To'] = config['receiver']

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(file, "rb").read())
    encoders.encode_base64(part)

    part.add_header('Content-Disposition', 'attachment; filename="report.csv"')

    msg.attach(part)

    ssl_context = ssl.create_default_context()
    service = smtplib.SMTP_SSL(config['server'], config['port'], context=ssl_context)
    service.login(config['user'], config['password'])

    result = service.sendmail(config['user'], config['receiver'], msg.as_string())

    service.quit()


if __name__ == '__main__':
    main()
