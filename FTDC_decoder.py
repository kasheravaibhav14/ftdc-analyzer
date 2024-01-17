import bson
import zlib
import struct
import io
from FTDC_analysis import FTDC_an
from datetime import datetime,timedelta
import ctypes
import os
import time
import argparse
from urllib.parse import urlparse
import string 
import secrets
import tarfile 
import requests 
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import shutil
from sprcommon import get_spr_env, get_group_var

# global variables for S3 and openAI keys which are set later in "__main__"
S3_BUCKET = "es-data-accumulator-bucket"
AWS_ACCESS_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
AWS_REGION = "ap-south-1"
OPENAI_API_KEY = "sk-"

def int64(uint64_value):
    # Create a ctypes unsigned 64-bit integer from the input value
    uint64_type = ctypes.c_uint64(uint64_value)
    # Convert the unsigned 64-bit integer to a signed 64-bit integer
    int64_type = ctypes.c_int64(uint64_type.value)
    # Return the signed 64-bit integer value
    return int64_type.value

def convert_to_datetime(datetime_str):
    '''
    Helps to parse the metrics filename as datetime

    Args:
        datetime_str(str): input date string
    Returns:
        datetime_ob(datetime)
    '''
    datetime_format = "%Y-%m-%dT%H-%M-%SZ-%f"
    datetime_obj = datetime.strptime(datetime_str, datetime_format)
    return datetime_obj


class FTDC:
    def __init__(self, metric_path,query_dt, outpath='',duration=300, exact=0):
        '''
        Initialises the FTDC object with the specified parameters.

        Args:
            metric_path (str): Path to the metric.
            query_dt (str): Query date.
            outpath (str, optional): Path to output. Defaults to ''.
            duration (int, optional): Duration in seconds. Defaults to 300 (5 minutes).
            exact (int, optional): If set to 1, performs an exact match. Defaults to 0.
        '''
        self.fpath=metric_path
        self.metric_names=[]
        self.prev_metric_list=[]
        self.metric_list={}
        self.metaDocs=[]
        self.rawDataDocs=[]
        self.tdelta=timedelta(hours=2, minutes=30)
        self.qTstamp=query_dt
        self.outpath=outpath
        self.duration=duration
        self.exact=exact

    def read_varuint(self,buf):
        '''
        Reads an unsigned integer from a buffer.

        Args:
            buf (Buffer): The buffer to read from.

        Returns:
            int: The read unsigned integer.'''
        value = 0
        shift = 0
        while True:
            b = buf.read(1)
            if not b:
                return -1
            byte = ord(b)
            value |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                break
        return value
    
    def __extract(self):
        '''
        Extracts metadata(type=0) and raw data(type=1) documents from the first level decoded items.
        '''
        for doc in self.items:
            if int(doc['type'])==1:
                self.rawDataDocs.append(doc)
            elif int(doc['type'])==0:
                self.metaDocs.append(dict(doc))
        
    def create_metric(self, data, prevkey=""):
        '''
        Recursively creates metrics from the provided data. This method traverses the data which can be a dictionary,
        a list, or a scalar value (such as an integer, a boolean, or a datetime). It generates metric names based on
        the keys of the dictionary or the indices of the list. For scalar values, it directly appends them to the metric
        list with their corresponding metric names. Special cases are handled for data types like 'bson.timestamp.Timestamp'
        and 'datetime', where they are converted to milliseconds epoch before storing. String and 'bson.objectid.ObjectId' 
        types are ignored.
        This is done according to mongodb's implementation of FTDC

        Args:
            data (dict or list or scalar): The input data from which to create the metrics. This data can be a nested 
                                        dictionary or list containing further dictionaries, lists or scalar values.
            prevkey (str, optional): The key name from the previous level of recursion. It is used to form the metric 
                                    name by concatenating it with the current level's key or index. Defaults to "".

        '''
        if isinstance(data, dict):
            for key,val in data.items():
                if prevkey=="":
                    nkey=key
                else:
                    nkey=prevkey+"."+key
                self.create_metric(val,nkey)
        elif isinstance(data, list):
            for idx,item in enumerate(data):
                nkey=prevkey+"._"+str(idx)
                self.create_metric(item,nkey)
        else:
            if type(data) == bson.timestamp.Timestamp:
                k0=prevkey
                k1=prevkey+".inc"
                self.metric_names.append(k0)
                self.metric_names.append(k1)
                self.metric_list[k0]=[data.time*1000]
                self.metric_list[k1]=[data.inc]

            elif type(data) == datetime:
                ms_epoch=data.timestamp()*1000
                self.metric_names.append(prevkey)
                self.metric_list[prevkey]=[ms_epoch]

            elif type(data) != str and type(data) != bson.objectid.ObjectId:
                self.metric_names.append(prevkey)
                if type(data) == bool:
                    data = int64(int(data))
                self.metric_list[prevkey]=[data]

    def parseBson(self,buf):
        size_data = buf.read(4)
        if len(size_data) < 4:
            return None  # Incomplete data, cannot form a valid BSON object

        size = struct.unpack("<i", size_data)[0]  # Unpack the size as a 32-bit signed integer
        data = size_data + buf.read(size - 4)  # Read the remaining bytes based on the size
        if len(data) < size:
            return None  # Incomplete data, cannot form a valid BSON object
        try:
            return bson.BSON.decode(data)  # Decode the BSON data
        except bson.errors.InvalidBSON:
            return None  # Invalid BSON data
        
    def __decodeData(self):
        '''
        This method decodes and processes data stored in BSON format, and compresses it with zlib compression.

        It iterates through each document in rawDataDocs. Each document is decompressed and parsed. If the parsed data's start 
        time is within a specified range, the method creates metrics from the parsed data.

        The method then unpacks and reads certain metrics from the reader, and stores each metric's values in the metric_list. 
        For each metric, the method also reads variable-length unsigned integer deltas and adds them to the base_val (first value 
        of each metric) to calculate the subsequent values. If a delta is zero, the method reads the next value as the number 
        of zeros to be inserted. This is followed to ensure run length encoding. 
        If the base value is a datetime object, the delta is added as seconds to the base_val. Otherwise, 
        the delta is directly added to the base_val.

        After all deltas are processed for a metric, the method stores the metric_list in the 
        accumulate_metrics dictionary. The metric_list and metric_names are then reset for the next document.

        If an exception occurs at any point, an error message is printed.

        Finally, the method instantiates an FTDC_an object and calls its parseAll method.

        Note:
        FTDC (Full Time Diagnostic Data Capture) is a method used in MongoDB for capturing and storing server status and other diagnostic data.

        Args:
        No arguments are required.

        Returns:
        None, but the method modifies the state of the object by setting various internal properties and instantiating 
        a FTDC_an object.
        '''
        accumulate_metrics={}
        ndet_tot=0
        for doc in self.rawDataDocs:
            try:
                to_decode=doc['data']
                decompressed_dat = zlib.decompress(to_decode[4:]) #first 4 bytes = header(uncompressed by default)
                reader=io.BytesIO(decompressed_dat)
                res=self.parseBson(reader)
                if abs(res['start']-self.qTstamp) > self.tdelta:
                    continue
                self.create_metric(res) # creates a metrics object by recursively flattening the BSON dict to a flattened key value pair
                stats=reader.read(8) # first 4 bytes correspond to the number of metrics, and the second correspond to num of deltas(total seconds - 1)
                nmetrics,ndeltas=(struct.unpack("<I", stats[0:4]),struct.unpack("<I", stats[4:8]))
                ndet_tot+=(ndeltas[0]+1)
                nzeros=0
                for met_idx in range(nmetrics[0]):
                    base_val=self.metric_list[self.metric_names[met_idx]][0] #the first document contains the base value, and the succeeding ones are only deltas encoding usinf run-length encoding
                    for del_idx in range(ndeltas[0]):
                        delta=0
                        if nzeros!=0: # if we have available zeros we will add them to metric value list before reading a new element
                            delta=0
                            nzeros-=1
                        else:
                            delta= self.read_varuint(reader)
                            if delta==-1:
                                raise ValueError("Error in Reading")
                            if delta == 0: # if delta value is 0, we read the next uint, to handle run length encoding
                                # in run-length encoding 0 0 0 0 0 is stored as 0 5, so we read 2 variables to add 5 0 points to our metric values
                                nzeros= self.read_varuint(reader)
                                if nzeros==-1:
                                    raise ValueError("Error in Reading")
                        if type(base_val)==datetime: 
                            base_val= base_val + timedelta(seconds=delta)
                        else:
                            base_val= base_val + int64(delta)
                        self.metric_list[self.metric_names[met_idx]].append(base_val)
                tstamp=datetime.utcfromtimestamp(self.metric_list['start'][0]/1000) # converting ms to utc timestamp
#                 tstamp=tstamp.strftime("%Y-%m-%d_%H-%M-%S")
                accumulate_metrics[tstamp]=self.metric_list
                self.metric_list={}
                self.metric_names=[]
                reader.close()
            except Exception as e:
                print("Failed to extract: ",e)
        # print(ndet_tot)
        tstamp=(next(iter(accumulate_metrics)))
        an_obj=FTDC_an(accumulate_metrics,self.qTstamp,self.outpath,self.duration,self.exact,OPENAI_API_KEY)
        an_obj.parseAll()

    def process(self):
        docs=[]
        for filepath in self.fpath: # iterate through files parsed by "__main__"
            with open(filepath,'rb') as file:
                data=file.read()
            docs.extend(bson.decode_all(data)) # first layer of decompression 
        self.items = docs
        self.__extract() # extract meta and data documents separately
        self.__decodeData() # decode the data documents

def validate_directory_or_file(value):
    """
    Validates whether the given value is a directory or a file.
    """
    if not os.path.isdir(value) and not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"Directory '{value}' does not exist")
    return value

def validate_directory(value):
    """
    Validates whether the given value is a directory. If not it creates it and returns the directory name
    """
    if not os.path.isdir(value):
        os.mkdir(value)
    return value

def validate_url(value):
    try:
        result = urlparse(value)
        if result.scheme and result.netloc:
            return value
        else:
            raise argparse.ArgumentTypeError('Invalid URL')
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid URL')

def validate_timestamp(value):
    """
    Validates whether the given value can be converted to a UTC timestamp.
    """
    # print(type(value))
    try:
        dtobj=datetime.utcfromtimestamp(int(value)//1000)
        return dtobj
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid timestamp, required format is YYYY-MM-DD HH:MM:SS')

def validate_interval(value):
    """
    Validates whether the given value is a positive integer and is valid for a bucket interval, default is 5 if not specified.
    """
    if not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError('Interval should be a positive integer')
    return int(value)

def find_files(directory, paths=[]):
    """
    Recursively finds all files in a given directory.
    """
    for entry in os.scandir(directory):
        if entry.is_file():
            paths.append(entry.path)
        elif entry.is_dir():
            find_files(entry.path, paths)
    return paths

def extract_files_from_tar(file_path, target_path):
    """
    Extracts all files from a tar file into a target directory.
    """
    extracted_files=[]
    # Check if the target_path directory exists
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with tarfile.open(file_path) as tar:
        for member in tar.getmembers():
            if member.isfile():  # check if it is a file
                # To extract only files, make sure the output filename does not include any directory structure
                filename = os.path.basename(member.name)
                member.name = filename  # reset the member name to just the filename
                tar.extract(member, path=target_path)  # extract the file
                extracted_files.append(target_path+"/"+member.name)
    return extracted_files


def download_file(url, destination):
    """
    Downloads a file from a given URL to a local destination.
    """
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Check if the request header contains the file size information
        file_size = response.headers.get('Content-Length')
        if file_size is not None:
            file_size = int(file_size)

        downloaded_size = 0
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=2**21):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    # If file size is known, we can calculate the progress
                    if file_size is not None:
                        print(f"Downloaded: {downloaded_size / 1024 / 1024:.2f}MB of {file_size / 1024 / 1024:.2f}MB", end='\r')
                    else:
                        print(f"Downloaded: {downloaded_size / 1024 / 1024:.2f}MB", end='\r')

        print(f"\nDownload finished. The file was saved to {destination}")
    else:
        print(f"Failed to download the file. Server responded with status code {response.status_code}.")

def upload_file_s3(filepath, key, args):
    '''
    Uploads a file to an AWS S3 bucket.

    The function first checks if the file exists and if the necessary environment variables are set. 
    Then, it creates an S3 resource object using the AWS credentials, and uploads the file to the specified 
    S3 bucket. After uploading, it creates an S3 client object and tries to retrieve the uploaded file 
    to ensure that it was uploaded successfully. If the file is uploaded, the function returns the file's 
    presigned URL which can be used to access the file. If the file wasn't found in the bucket or if there 
    were no credentials found, the function returns an appropriate error message.

    Args:
        filepath (str): The local path to the file to be uploaded.
        key (str): The name that will be used for the file in the S3 bucket.

    Returns:
        str: The presigned URL of the uploaded file if the upload was successful, 
             or an error message if the upload failed.

    Raises:
        ValueError: If the provided file path does not point to a file.
    '''
    global SPR_ENV_NAME, AWS_REGION, AWS_ACCESS_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
    # validate the file path
    if not os.path.isfile(filepath):
        raise ValueError("The file does not exist or is not a file: {}".format(filepath))

    # validate and retrieve environment variables
#     if "AWS_REGION" not in os.environ or "AWS_ACCESS_KEY_ID" not in os.environ or "AWS_SECRET_ACCESS_KEY" not in os.environ or "BUCKET_NAME" not in os.environ:
#         raise EnvironmentError("Required environment variables are not set")

    if "AWS_REGION" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ and "BUCKET_NAME" in os.environ:
        AWS_REGION = os.getenv('AWS_REGION')
        AWS_ACCESS_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        S3_BUCKET = os.getenv('BUCKET_NAME')
    
    elif args.aws_region is not None and args.s3_bucket_name is not None and args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        AWS_REGION = args.aws_region
        AWS_ACCESS_ID = args.aws_access_key_id
        AWS_SECRET_ACCESS_KEY = args.aws_secret_access_key
        S3_BUCKET = args.s3_bucket_name
        
    else:
        AWS_ACCESS_ID = get_group_var(SPR_ENV_NAME, varname="vf_prod_s3_access_key")
        AWS_SECRET_ACCESS_KEY = get_group_var(SPR_ENV_NAME, varname="vf_prod_s3_secret_key")
#         AWS_REGION = get_group_var(SPR_ENV_NAME, varname="us-east-1")
        AWS_REGION = "us-east-1"
#         S3_BUCKET = get_group_var(SPR_ENV_NAME, varname="ht_s3_backup_bucket")
        S3_BUCKET = "genericbackup.sprinklr.com"
    try:
        # Creating a resource for 's3' 
        s3_resource = boto3.resource(
            's3', 
            region_name = AWS_REGION,
            aws_access_key_id = AWS_ACCESS_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        ) 

        # Upload a file to S3 bucket
        s3_resource.Bucket(S3_BUCKET).put_object(
            Key = key, 
            Body = open(filepath, 'rb')
        )

        # Creating a client for 's3' 
        s3_client = boto3.client(
            's3', 
            region_name = AWS_REGION,
            aws_access_key_id = AWS_ACCESS_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )
        try:
            # Check if the file was uploaded successfully
            s3_client.head_object(Bucket=S3_BUCKET, Key=key)
            print("File was uploaded successfully")
            # Generate the URL to get 'key-name' from 'bucket-name'
            url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': key
                }
            )
            return url

        except ClientError:
            # The file wasn't found. 
            return "File was not found in the bucket. Upload failed."
    except NoCredentialsError:
        return "No AWS credentials were found"

def generate_random_string(length=5):
    '''
    Generate a randomly generated string
    Args:
        length(optional, default=5)
    '''
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(length))

def get_open_ai_key(args):
    '''
    Get openAI API key from searching under environment, commandline args and then sprinklr servers '''
    global OPENAI_API_KEY
    if "OPENAI_API_KEY" in os.environ:
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    elif args.openai_api_key is not None:
        OPENAI_API_KEY=args.openai_api_key
    else:
        OPENAI_API_KEY=get_spr_env(SPR_ENV_NAME,"ht_openai_key")

if __name__ == "__main__":
    '''
    sample runner:
    python3 FTDC_decoder.py --input_url https://awsbucket-147.s3.ap-northeast-1.amazonaws.com/prod2-mongo-schedulerv7-9-z2.10.75.16.7.diagnostic.data.tar.gz --timestamp 1687871400000 --output_dir mydir1
    
    This is the entry point of the script execution. It gets the environment, generates a random output file name,
    sets up command-line argument parsing, handles the input data either from a local directory or a URL, handles 
    AWS related information, sets the timestamp to process the data, the output file path, the interval for data 
    aggregation, and whether to use exact time stamp or not. It also manages file downloading, extraction and filtering.
    Finally, it processes the FTDC data and uploads the resulting file to S3 if necessary, then cleans up the temporary 
    files and directories.

    Args:
        --input: Tar path or extracted directory path.
        --input_url: Input URL.
        --aws_region: AWS region.
        --s3_bucket_name: Name of the S3 bucket.
        --aws_access_key_id: AWS access key ID.
        --aws_secret_access_key: AWS secret access key.
        --openai_api_key: OpenAI API key.
        --timestamp: Milliseconds from epoch.
        --output: Output file path.
        --output_dir: Output directory to save the report and temporary extracted files
        --interval: Interval in minutes for bucket duration.
        --exact: Integer indicating whether to use the exact timestamp as the drop time or not.

    Returns:
        None. The output of the script is the generated PDF report and the optional upload of the report to an AWS S3 bucket.

    '''
    SPR_ENV_NAME = get_spr_env()
    outFile = "report-"+generate_random_string()+".pdf" #just in case the tar file has no name
    parser = argparse.ArgumentParser(description="""FTDC Decoder Script. 
    This script accepts either a tar/directory of input files or a URL to fetch the input data. 
    It also requires a timestamp(ms from epoch), and an output file path to the pdf.
    The interval parameter determines the bucket period in minutes. 
    An 'exact' parameter can be set to 1, if we do not want to search for a ticket and assume there is a drop ticket present at the given timestamp.""")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=validate_directory_or_file, help="tar path or extracted directory path")
    input_group.add_argument("--input_url", type=validate_url, help="Input URL")
    
    
    parser.add_argument("--aws_region", help="aws region", required=False)
    parser.add_argument("--s3_bucket_name", help="name of s3 bucket", required=False)
    parser.add_argument("--aws_access_key_id", help="access key id of aws", required=False)
    parser.add_argument("--aws_secret_access_key", help="secret access key of aws", required=False)
    parser.add_argument("--openai_api_key", help="secret access key of aws", required=False)
    parser.add_argument("--output_dir", type=validate_directory, help="output directory for report and temp files", default=".", required=False)

    parser.add_argument("--timestamp", type=validate_timestamp, required=True, help="Milliseconds from epoch")
    parser.add_argument("--output", type=str, required=False,default=outFile, help="Output file path")
    parser.add_argument("--interval", type=validate_interval, required=False, default=5, help="Interval in minutes")
    parser.add_argument("--exact", type=int, required=False, default=0, help="set the timestamp as drop time t0")

    args = parser.parse_args()
    print("Requested Query Timestamp:",args.timestamp)
    destination=""
    tarfilename=""

    if args.input_url: # if input file is a URL
        file_url = args.input_url
        try:
            tarfilename=file_url[file_url.rindex('/')+1:file_url.index('.tar')]
        except ValueError as e:
            print(e, "Check if the download url is correct and points to a tar file")
            exit(1)
        file_name = generate_random_string(3)+"_"+tarfilename+file_url[file_url.index('.tar'):]
        download_file(file_url,file_name)
        destination=file_name
    elif args.input and os.path.isfile(args.input): # if input file is local and a tarfile, not an extracted direcctory
        destination=args.input
        try:
            tarfilename=destination[destination.rindex('/')+1:destination.index('.tar')]
        except ValueError as e:
            print(e, "Not a tar file or bad extension")
            exit(1)

    if destination != "": # check if we are dealing with tar file and not extracted files in a directory
        out_folder = generate_random_string(10) # generate a name for temporary directory inside a specified directory
        out_folder = os.path.join(args.output_dir,out_folder)
        print(out_folder)
        os.mkdir(out_folder)
        files=extract_files_from_tar(destination,out_folder)
        if args.input_url:
            os.remove(destination)
    else:
        files=find_files(args.input)

    # st=time.time()
    get_open_ai_key(args) # extract open ai key in the global variable for chatgpt-4 inference
    filtered_files=[]
    all_files= [i for i in files if "metrics" in i] # check to only process metrics file and not any other file in the directory
    all_files.sort() # sort by name and hence timestamp
    # print(all_files)
    for file in all_files:
        file_name=file[file.rindex('/')+1:]
        if "interim" not in file_name: # process all files except metrics.interim, handled below
            tstamp=convert_to_datetime(file_name[file_name.index('.')+1:])
            diff=abs(tstamp-args.timestamp)
            if diff <= timedelta(hours=6): # capture all files within a 6 hour delta of the requested timestamp.
                filtered_files.append(file)
    filtered_files.sort()
    if len(filtered_files)==0:
        raise ValueError("No files corresponding to the queryTimestamp found. Please check the timestamp/path and try again!")
    
    if all_files.index(filtered_files[-1]) == len(all_files)-2: # metrics.interim should be included
        filtered_files.append(all_files[-1])
    
    report_filename = "report-"+tarfilename+".pdf"
    report_filename = os.path.join(args.output_dir, report_filename)
    try:
        decoder = FTDC(filtered_files,args.timestamp,report_filename,args.interval*60, args.exact)
        decoder.process()
        # st=time.time()-st
        if os.path.isfile(report_filename):
            print("report locally saved as:",report_filename)
            downloadUrl = upload_file_s3(report_filename,report_filename, args)
            print(downloadUrl)
    except Exception as e:
        print(e)
    finally: # remove the folder in which tar files are extracted after everything is done
        if destination!="":
            shutil.rmtree(out_folder)