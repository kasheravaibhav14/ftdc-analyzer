import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import requests
import time
from FTDC_plot import FTDC_plot

class FTDC_an:
    """
    Description:
    This class is designed to handle analysis of MongoDB's FTDC data which includes 
    the preparation of certain metrics for visualizing and processing the data.
    The main responsibilites of this class include structuring the data and performing analysis to filter only relevant metrics.
    The class also creates an input string to request response from chatgpt-4
    Then it renders a plot in a PDF integrated with output of chatgpt-4

    Attributes:
        metricObj (dict): A dictionary that contains the metrics to be analyzed.
        queryTimeStamp (str): A string representing the timestamp of the query.
        ticketlim (int): An integer that sets the thresold for ticket drop. It is initialized to 50.
        tdelta (int): The time duration for which the analysis will be performed.
        outPDF (str): The path of the output PDF file where the result of the analysis will be stored.
        nbuckets (int): The number of buckets to be used in the analysis. It is initialized to 12.
        anomalyBuckets (int): The number of buckets that contain anomalies. It is initialized to 3.
        meanThreshold (float): The threshold for the mean of anomalies. It is initialized to 1.25.
        totalTickets (int): The total number of tickets available. It is initialized to 128.
        exact (bool): A boolean flag that determines whether the time is exact or not.
        openai_api_key (str): The API key to be used for OpenAI's GPT-4 model.
    """
    def __init__(self, metricObj, qTstamp, outPDFpath, duration, exact, openai_api_key):
        self.metricObj = metricObj
        self.queryTimeStamp = qTstamp
        self.ticketlim = 50
        self.tdelta = duration
        self.outPDF = outPDFpath
        self.nbuckets = 12
        self.anomalyBuckets = 3
        self.meanThreshold = 1.25
        self.totalTickets = 128
        self.exact = exact
        self.openai_api_key = openai_api_key
        self.openai_chat_completition_endpoint = "https://api.openai.com/v1/chat/completions"

    def __plot(self, df, to_monitor, vert_x, gpt_out=""):
        """
        Description:
        This method plots the metrics data onto a PDF file.

        Args:
        df (pd.DataFrame): The DataFrame containing the metrics data.
        to_monitor (list): A list of metrics to monitor.
        vert_x (list): The x-coordinates for the vertical lines in the plot.
        gpt_out (str, optional): Text output from GPT model. Default is "".

        Returns: 
        None. The function saves the plot to a PDF file and does not return any value.
        """
        to_monitor.sort()
        print(len(to_monitor), "metrics")
        plot_ob = FTDC_plot(df, to_monitor, vert_x, gpt_out, self.outPDF)
        plot_ob.plot()
        return

    def __renameCols(self, df):
        """
        Description:
        This method renames the columns of the dataframe to a more readable format.

        Args:
        df (pd.DataFrame): The DataFrame containing the metrics data.

        Returns: 
        None. The function renames the DataFrame's columns in-place and does not return any value.
        """
        rename_cols = {}
        for col_name in df.columns:
            ccol = col_name.replace("serverStatus.", "ss ")
            ccol = ccol.replace("wiredTiger.", "wt ")
            ccol = ccol.replace("systemMetrics.", "sm ")
            ccol = ccol.replace("tcmalloc.tcmalloc", "tcmalloc")
            ccol = ccol.replace("transaction.", "txn ")
            ccol = ccol.replace("local.oplog.rs.stats.", "locOplogRsStats ")
            ccol = ccol.replace("aggStageCounters.", "aggCnt ")
            rename_cols[col_name] = ccol
        df.rename(columns=rename_cols, inplace=True)

    def getDTObj(self, date_string):
        """
        Description:
        This method converts a date string into a datetime object.

        Args:
        date_string (str): The date string in the format "%Y-%m-%d_%H-%M-%S".

        Returns: 
        A datetime object.
        """
        format_string = "%Y-%m-%d_%H-%M-%S"
        parsed_datetime = datetime.strptime(date_string, format_string)
        return parsed_datetime

    def getDTFromMilliseconds(self, ms):
        """
        Description:
        This method converts milliseconds since epoch into a datetime object.

        Args:
        ms (int): Milliseconds since epoch.

        Returns: 
        A datetime object.
        """
        return datetime.fromtimestamp(ms/1000)

    def __getDirtyFillRatio(self, metricObj):
        """
        Description:
        This method calculates the ratio of dirty bytes to total bytes in the cache.

        Args:
        metricObj (dict): The dictionary containing the metrics data.

        Returns: 
        None. The function modifies the given dictionary in-place to include the dirty fill ratio.
        """
        total_cache = metricObj["serverStatus.wiredTiger.cache.bytes currently in the cache"]
        dirty_cache = metricObj["serverStatus.wiredTiger.cache.tracked dirty bytes in the cache"]
        metricObj["ss wt cache dirty fill ratio"] = []
        for idx in range(len(total_cache)):
            if total_cache[idx] != 0:
                ratio = (dirty_cache[idx] / total_cache[idx])
            else:
                ratio = 0
            metricObj["ss wt cache dirty fill ratio"].append(100*ratio)

    def __getCacheFillRatio(self, metricObj):
        """
        Description:
        This method calculates the ratio of current bytes to total bytes in the cache.

        Args:
        metricObj (dict): The dictionary containing the metrics data.

        Returns: 
        None. The function modifies the given dictionary in-place to include the cache fill ratio.
        """
        total_cache = metricObj["serverStatus.wiredTiger.cache.maximum bytes configured"]
        curr_cache = metricObj["serverStatus.wiredTiger.cache.bytes currently in the cache"]
        metricObj["ss wt cache fill ratio"] = []

        for idx in range(len(total_cache)):
            if total_cache[idx] != 0:
                ratio = (curr_cache[idx] / total_cache[idx])
            else:
                ratio = 0
            metricObj["ss wt cache fill ratio"].append(100*ratio)

    def __getMemoryFragRatio(self, metricObj):
        """
        Description:
        This method calculates the memory fragmentation ratio.

        Args:
        metricObj (dict): The dictionary containing the metrics data.

        Returns: 
        None. The function modifies the given dictionary in-place to include the memory fragmentation ratio.
        """

        tCache = "serverStatus.tcmalloc.generic.current_allocated_bytes"
        trCache = "serverStatus.tcmalloc.generic.heap_size"
        nkey = "serverStatus.wiredTiger.memory fragmentation ratio"
        if trCache not in metricObj or tCache not in metricObj:
            return
        metricObj[nkey] = []
        for idx in range(len(metricObj[trCache])):
            if metricObj[trCache][idx] != 0:
                metricObj[nkey].append(
                    100*((metricObj[trCache][idx]-metricObj[tCache][idx])/metricObj[trCache][idx]))
            else:
                metricObj[nkey].append(-1)

    def __getAverageLatencies(self, metricObj):
        """
        Description:
        This method calculates the average latency for various commands.

        Args:
        metricObj (dict): The dictionary containing the metrics data.

        Returns: 
        None. The function modifies the given dictionary in-place to include the average latencies.
        """
        base = "serverStatus.opLatencies."
        for command in ["reads.", "writes.", "commands.", "transactions."]:
            opkey = base+command+"ops"
            ltkey = base+command+"latency"
            if opkey in metricObj:
                for idx in range(len(metricObj[opkey])):
                    if metricObj[opkey][idx] != 0:
                        metricObj[ltkey][idx] = metricObj[ltkey][idx] / \
                            metricObj[opkey][idx]

    def __diskUtilization(self, metricObj):
        """
        Description:
        This method calculates the disk utilization.

        Args:
        metricObj (dict): The dictionary containing the metrics data.

        Returns: 
        None. The function modifies the given dictionary in-place to include the disk utilization.
        """
        disks = []
        for key in metricObj:
            if key.startswith("systemMetrics.disks"):
                mystr = key
                disk = mystr.split("systemMetrics.disks.")[1].split('.')[0]
                if disk not in disks:
                    disks.append(disk)

        for disk in disks:
            io = "systemMetrics.disks."+disk+".io_time_ms"
            newkey = "systemMetrics.disks."+disk+" utilization%"
            if io not in metricObj:
                continue
            metricObj[newkey] = []
            for idx in range(len(metricObj[io])):
                if metricObj[io][idx] == 0:
                    metricObj[newkey].append(0)
                else:
                    metricObj[newkey].append(
                        (metricObj[io][idx])/(10))

    def __tcmallocminuswt(self, metricObj):
        """
        Description:
        This method calculates the difference between the memory allocated by tcmalloc and wiredtiger cache.

        Args:
        metricObj (dict): The dictionary containing the metrics data.

        Returns: 
        None. The function modifies the given dictionary in-place to include the memory difference.
        """
        wtcache = "serverStatus.wiredTiger.cache.bytes currently in the cache"
        tcmalloc = "serverStatus.tcmalloc.generic.current_allocated_bytes"
        newkey = "serverStatus.wiredTiger.tcmalloc derived: allocated minus wt cache MiB"
        if wtcache not in metricObj or tcmalloc not in metricObj:
            return
        itr = 0
        mib_conv = 2**20
        itr += 1
        metricObj[newkey] = []
        for idx in range(len(metricObj[wtcache])):
            metricObj[newkey].append(
                (metricObj[tcmalloc][idx]-metricObj[wtcache][idx])/mib_conv)
            
    def calcBounds(self, df, pos, delt):
        '''
        Calculate the bounds of a range within a DataFrame based on certain conditions.

        The function scans the DataFrame within a certain range, defined by the index of query timestamp(`pos`) and and interval duration.
        `delt` represents half of the time interval of a bucket
        It checks for conditions where read and write tickets fall below a specified limit. If both, 
        write or read ticket counts fall below the limit, it sets the starting point 't0' and type 'typ' 
        of the drop accordingly. 
        
        If no such conditions are met it sets 't0' to 'pos'. 
        If check is specified as 1, we ovveride the search process and set `t0` to `pos` instead.
        It then fills `tbounds` with the indices of the start of each interval in the dataframe. 
        Two consecutive values in `tbounds` can be used to slice an interval equal to the duration of the bucket.

        Args:
            df (pandas.DataFrame): The DataFrame to scan for ticket drops.
            pos (int): The center of the range to scan for ticket drops.
            delt (int): The half-width of each bucket, in the same time units as the DataFrame index.

        Returns:
            tbounds (list): A list of DataFrame indices bounding the intervals leading to where ticket drops were found. The last bucket is the ticket drop itself.
            t0 (int): The index of the DataFrame where the first ticket drop was found.
            typ (int): The type of the first ticket drop found: 0 for both, 1 for write, 2 for read, -1 if none.
        '''
        tbounds = []
        t0 = -1
        typ = -1
        pos1 = max(0,pos-self.tdelta*2)
        pos2 = min(pos+self.tdelta*6,len(df)-1)
        read_ticket = 'ss wt concurrentTransactions.read.available'
        write_ticket = 'ss wt concurrentTransactions.write.available'
        for idx in range(pos1, pos2):
            if df.iloc[idx][write_ticket] < self.ticketlim and df.iloc[idx][read_ticket] < self.ticketlim:
                t0 = idx
                typ = 0
                print("found both read and write ticket drop at: ",df.index[t0])
                break
            if df.iloc[idx][write_ticket] < self.ticketlim:
                t0 = idx
                typ = 1
                print("found write ticket drop at:", df.index[t0])
                break
            if df.iloc[idx][read_ticket] < self.ticketlim:
                t0 = idx
                typ = 2
                print("found read ticket drop at:", df.index[t0])
                break
        # print(t0)
        if typ == -1 or self.exact == 1:
            t0=pos
            print("Setting the ticket drop to:",df.index[pos],"as requested")
        idx = t0+delt
        # for i in range(0, 2): # one extra bucket ahead if available
        while (not df.index[idx] and idx < len(df)):
            idx += 1
        tbounds.append(idx)
        for i in range(0, self.nbuckets):
            if idx <=0:
                break
            idx -= 2*delt
            while (not df.index[idx] and idx > 0):
                idx -= 1
            tbounds.insert(0, idx)
        return tbounds, t0, typ

    def has_outliers(self, data):
        """
        This method checks if a given list contains any outliers.

        An outlier is defined as a value that is below Q1 - `meanThreshold` * IQR or above Q3 + `meanThreshold` * IQR.
        Where Q1 and Q3 are the first and third quartiles of the data, and IQR is the interquartile range (Q3 - Q1).

        If a value is found in the last `anomalyBuckets` number of elements in the list,
        then that value is returned along with its position. If no such value is found,
        False is returned along with the count of data.

        Parameters:
        data (list of int/float): The data to check for outliers.

        Returns:
        tuple: A tuple containing a boolean indicating if an outlier was found and its position or count of data.
        """
        multiplier = self.meanThreshold
        Q1 = np.percentile(data, 25)  # First quartile (Q1)
        Q3 = np.percentile(data, 75)  # Third quartile (Q3)
        IQR = Q3 - Q1  # Interquartile range (IQR)

        # Deviation for considering a value as an outlier
        dev = multiplier * IQR

        # Compute lower and upper bounds for outliers
        lower_bound = Q1 - dev
        upper_bound = Q3 + dev

        ctr = 0  # counter to keep track of the current position in the list
        curr_ctr = 0  # counter to keep track of the position of the current outlier
        curr_val = None  # variable to store the current outlier

        # Loop over the data to find outliers
        for value in data:
            if value < lower_bound or value > upper_bound:  # If the value is an outlier
                if curr_val == None:  # If this is the first outlier
                    curr_val = value
                    curr_ctr = ctr
                elif ctr > curr_ctr:  # If this outlier is found after the current outlier
                    curr_val = value
                    curr_ctr = ctr
            ctr += 1  # Increase the position counter

        # If an outlier was found in the last 'anomalyBuckets' number of elements
        if curr_val != None and curr_ctr >= len(data)-self.anomalyBuckets:
            return True, curr_ctr  # Return True indicating an outlier was found and its position

        return False, ctr  # Return False indicating no outlier was found and the count of data

    def percentileChange(self, data, cont=0.2):
        """
        Description:
        This method uses Isolation Forest algorithm to identify outliers in the given data and checks if the last outlier is within the last three elements.

        Args:
        data (list or array-like): The input data in which to find outliers.
        cont (float, optional): The contamination factor for the Isolation Forest algorithm. Defaults to 0.2.

        Returns: 
        bool: True if the last outlier is within the last three elements, False otherwise.
        int: The index of the last outlier, or 0 if there are no outliers.
        """
        # Ensure data is a numpy array and reshape it.
        data = np.array(data).reshape(-1, 1)

        # Initialize and fit the IsolationForest model.
        iso_forest = IsolationForest(contamination=cont)
        iso_forest.fit(data)
        pred = iso_forest.predict(data)
        outliers = data[pred == -1]

        # Check if there are no outliers.
        if len(outliers) == 0:
            return False, 0

        # Find the index of the last outlier.
        last_outlier_index = np.where(data == outliers[-1])[0][0]

        # Check if the last outlier is within the last three elements.
        if last_outlier_index >= len(data)-self.anomalyBuckets:
            return True, last_outlier_index

        return False, 0

    def calculate_statistics(self, data):
        """
        Description:
        This method calculates the mean and the 99th percentile of the input data.

        Args:
        data (list or array-like): The input data for which to calculate statistics.

        Returns: 
        float: The mean of the input data.
        float: The 99th percentile of the input data.
        """
        return np.mean(data), np.percentile(data, 99)

    def is_mean_stable(self, mean_values, perc_values):
        """
        Description:
        This method checks if the means of the input arrays are stable. The mean is considered stable if all elements of the input arrays are within the 'meanThreshold' of the overall mean.

        Args:
        mean_values (list or array-like): The first array of mean values.
        perc_values (list or array-like): The second array of percentile values.

        Returns: 
        bool: True if the mean is stable, False otherwise.
        """
        if np.mean(mean_values) == 0 or np.mean(perc_values) == 0:
            return True
        if all(abs(m / np.mean(mean_values)) <= self.meanThreshold for m in mean_values) and all(abs(m / np.mean(perc_values)) <= self.meanThreshold for m in perc_values):
            return True
        return False

    def check_metric(self, df, met, timebounds, containment=0.2):
        """
        Analyzes a specific metric in a pandas DataFrame within a specific time window.

        Args:
            df (DataFrame): Input pandas DataFrame.
            met (str): The metric to be analyzed.
            timebounds (list): Time boundaries.

        Returns:
            tuple: Tuple containing a boolean indicating whether a condition has been met,
                a bucket or index, and a tuple of statistics.
        """

        p99 = []
        means = []
        maxes = []
        for t in range(0, len(timebounds) - 1):
            temp_values = df.iloc[timebounds[t]:timebounds[t + 1]][met].values
            mean, percentile_99 = self.calculate_statistics(temp_values)
            p99.append(percentile_99)
            means.append(mean)
            maxes.append(np.max(temp_values))
            
        special_metrics = {"ss wt cache fill ratio": 80,
                           "ss wt cache dirty fill ratio": 5}
        special_metrics_perc = {"ss wt concurrentTransactions.read.out": self.totalTickets-self.ticketlim, "ss wt concurrentTransactions.write.out": self.totalTickets-self.ticketlim}
        if self.is_mean_stable(means, p99) and not (met in special_metrics_perc) and (met not in special_metrics):
            return False, 0, ()
        if met in special_metrics:
            indices = [index for index, element in enumerate(p99) if (element >= special_metrics[met]*0.95 or maxes[index] >= special_metrics[met]) and element >= np.mean(p99)]
            if indices:
                _idx = max(indices)
                return True, _idx, (means[_idx], np.mean(means), p99[_idx], np.mean(p99))
            return False, 0, ()
        if met in special_metrics_perc: # if 99 percentile of out tickets is more than 78(50 tickets remaining)
            indices = [index for index, element in enumerate(
                p99) if element > special_metrics_perc[met]]
            if indices:
                return True, max(indices), (means[max(indices)], np.mean(means), p99[max(indices)], np.mean(p99))
            else:
                return False, 0, ()

        tr, idx = self.has_outliers(means)
        tr1, idx1 = self.percentileChange(p99, cont=containment)

        if tr and tr1:
            return True, idx, (means[idx], np.mean(means), p99[idx], np.mean(p99))
        elif tr:
            return True, idx, (means[idx], np.mean(means), p99[idx], np.mean(p99))
        elif tr1:
            return True, idx1, (means[idx1], np.mean(means), p99[idx1], np.mean(p99))
        return False, 0, ()

    def _init_analytics(self, metricObj):
        """
        Description:
        This method will initialize analytics on the provided metrics object, adding derived metrics to the metricObj.

        Args:
        metricObj (object): The object containing the metrics to be analyzed.

        Returns: 
        None
        """
        # self.__getAverageLatencies(metricObj)
        self.__tcmallocminuswt(metricObj)
        self.__getMemoryFragRatio(metricObj)
        self.__getDirtyFillRatio(metricObj)
        self.__getCacheFillRatio(metricObj)
        self.__diskUtilization(metricObj)

    def _prepare_dataframe(self, metricObj):
        """
        Description:
        This method prepares a pandas DataFrame from the given metrics object. It applies various transformations and sets appropriate column names.

        Args:
        metricObj (object): The object containing the metrics to be transformed into a DataFrame.

        Returns: 
        df (pandas.DataFrame): The prepared DataFrame.
        """
        df = pd.DataFrame(metricObj)
        df['serverStatus.start'] = df['serverStatus.start'].apply(
            self.getDTFromMilliseconds)
#         df['serverStatus.start'] = pd.to_datetime(df['serverStatus.start'])
        df.drop(index=0)
        df.set_index('serverStatus.start', inplace=True)
        df.columns.name = 'metrics'
        self.__renameCols(df)
        # print(df)
        return df

    def _calculate_anomalies(self, df, tbounds, to_monitor):
        """
        Description:
        This method calculates anomalies for the given DataFrame and returns an anomaly map and a list of metrics to monitor.

        Args:
        df (pandas.DataFrame): The DataFrame for which to calculate anomalies.
        tbounds (list): A list containing the bounds for anomaly calculation.
        to_monitor (list): A list of metrics to monitor.

        Returns: 
        dict: The anomaly map.
        list: The updated list of metrics to monitor.
        """
        anomaly_map = {}
        for metric in df.columns:
            try:
                tr, idx, val = self.check_metric(df, metric, tbounds)
                if tr and not (metric.startswith("sm disks") and metric.endswith("io_time_ms")):
                    to_monitor.append(metric)
                    anomaly_map = self._update_anomaly_map(
                        metric, idx, val, anomaly_map)
            except Exception as e:
                print(e)
                print("unable to insert metric:", metric)
        return anomaly_map, to_monitor

    def _update_anomaly_map(self, metric, idx, val, anomaly_map):
        """
        Description:
        This method updates the given anomaly map with the provided metric, index, and value.

        Args:
        metric (str): The metric for which to update the anomaly map.
        idx (int): The index at which to update the anomaly map.
        val (various types): The value with which to update the anomaly map.
        anomaly_map (dict): The anomaly map to be updated.

        Returns: 
        dict: The updated anomaly map.
        """
        def compare_strings(lst): # sorting for getting prompt better
            s=lst[0]
            if s.startswith("ss wt concurrentTransactions."):
                return (0, s)
            elif s.startswith("ss metrics"):
                return (1, s)
            elif s.startswith("ss opcounters"):
                return (2, s)
            elif s.startswith("ss wt cache"):
                return (3, s)
            elif s.startswith("ss wt"):
                return (4, s)
            else:
                return (6, s)
        if idx not in anomaly_map:
            anomaly_map[idx] = []
        anomaly_map[idx].append([metric, val[0], val[1], val[2], val[3]])
        anomaly_map[idx].sort(key=compare_strings)
        return anomaly_map

    def _create_anomaly_obj(self, sorted_keys, anomaly_map, tbounds, df):
        """
        Description:
        This method creates an anomaly object based on the given sorted keys, anomaly map, bounds, and DataFrame.

        Args:
        sorted_keys (list): A list of sorted keys.
        anomaly_map (dict): The anomaly map from which to create the anomaly object.
        tbounds (list): A list containing the bounds for the anomalies.
        df (pandas.DataFrame): The DataFrame from which to create the anomaly object.

        Returns: 
        dict: The created anomaly object.
        """
        anomalyObj = {}
        for ts in sorted_keys:
            tsss = str(df.index[tbounds[ts]])
            if tsss not in anomalyObj:
                anomalyObj[tsss] = []
            for val in anomaly_map[ts]:
                anomalyObj[tsss].append({
                    "metric": val[0],
                    "anomaly interval mean": val[1],
                    "overall mean": val[2],
                    "anomaly interval 99th percentile": val[3],
                    "overall mean 99th percentile": val[4]
                })
        return anomalyObj

    def _create_gpt_str_base(self, df, t0, typ):
        ticket_type = "write" if typ == 1 else "read"
        if typ == 0:
            ticket_type = "both read and write"
        gpt_str_base = f'''You are a mongodb diagnostic engine specialising in determining the root cause of anomalous metrics provided to you. The given mongodb server has seen a drop in available {ticket_type} tickets at Timestamp {df.index[t0]}.During this time period, there were no significant changes in the server's hardware or software configuration. A "ticket drop" in this scenario signifies a rise in either concurrentTransactions.write.out or concurrentTransactions.read.out, typically due to lengthy operations not releasing the ticket promptly or an influx of concurrent server requests. Each operation acquires a ticket and releases it after the task is done.

TASK: Your task, as a MongoDB diagnostic specialist, is to analyze the given data with reference to MongoDB and WiredTiger engine metrics to determine the ticket drop's root cause. Please analyze each and every metric listed in the list provided.

Important thresholds and information include:
1. Start with analyzing ss metrics commands, operation, queryExecutor, etc. and opCounters (updates, deletes etc.). Any surge in any of these is indicative of increase in workload, which could be a potential indicator of ticket drop and must be included in analysis. 
2. Examine cache dirty/fill ratios. When cache dirty ratio surpasses 5%, eviction is initiated by worker threads and on crossing 20%, by application threads. A cache fill ratio over 80% initiates worker thread eviction and above 95% starts application thread eviction.
3. Reviewing eviction statistics due to their impact on worker threads and cache. Remember that evicting a modified page demands more resources.
4. Check 'cursor.cached cursor count', a measure of currently cached cursors by the WiredTiger engine.
5. Monitor logicalSessionRecordCache, used by MongoDB to track client sessions status.
6. Review disk utilization values. High values can indicate disk bottleneck. Anything below 50% can be safely ignored.

These pointers should supplement your analysis, not limit it. As a specialist, interpret each metric and its implications.

Note: Always examine percentile values for cache dirty and fill ratios, and be alert for any anomalies, especially in opCounters and ss metrics (commands, operation, queryExecutor). Since we are dealing with intervals, a looking at both mean and 99th percentile would give you a better insight.

Abbreviations to note:
'sm' - system metrics, 'ss' - server status, 'wt' - wiredtiger.

Data Format:
Each timestamp denotes the interval from itself to {self.tdelta//60} minutes ahead of it. For example, anomaly interval mean at timestamp t, means the mean of the given metric in [t,t+{self.tdelta//60} minutes]. 

The data contains timestamps and a list of anomalous metrics that were anomalous in the interval denoted by the timestamp. The meaning of each heading is as follows:
`anomaly interval mean`: mean of the metric in the timestamp interval where it was anomalous 
`overall mean`: mean of the metric over the monitored duration
`anomaly interval 99th percentile`: 99th percentile value of the metric in the timestamp interval where it was anomalous 
`overall mean 99th percentile`: mean of 99th percentile value of all intervals in the monitored duration 

Output Format: Provide a well-structured and comprehensive summary first and then a deeper detailed explanation of your analysis. Make sure no crucial details or metrics are overlooked. Every place you mention a timestamp, use "In the interval between <Timestamp> and <Timestamp+{self.tdelta//60}> ...."

NOTE: The focus is on in-depth analysis, so please refer to definitions and detailed implications of each metric as needed from your model.
'''
        return gpt_str_base

    def _save_gpt_str_base(self, gpt_str_base):
        with open("gpt-input.txt", 'w') as gpt:
            gpt.write(gpt_str_base)

    def _openAI_req(self, message):
            url = self.openai_chat_completition_endpoint
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": message}]
            }
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()  # This will raise an exception for HTTP errors
                completion = response.json()
                outp = completion['choices'][0]['message']['content']
                print(outp)
            except Exception as e:
                print(e)
                print("Generating report without summary as openAI failed to respond.")
                outp = ""
            return outp

    def analytics(self, metricObj, queryTimestamp):
        """
        Description:
        This method initializes analytics based on the metric object, prepares a dataframe from the metric object, calculates bounds, and then checks if there is an anomaly.
        If an anomaly is found, it calculates the anomaly, creates a gpt string base, creates an anomaly object, sends a request to OpenAI's GPT model, and finally generates a plot.
        
        Args:
        metricObj (dict): The metrics object that will be used for initializing analytics and preparing the dataframe.
        queryTimestamp (str): The timestamp for which the analytics are being run.
        
        Returns:
        None

        Note:
        This method does not return any value but modifies the internal state of the object by calculating anomalies, creating gpt string, and generating plots based on the processed data.
        """
        self._init_analytics(metricObj)
        df = self._prepare_dataframe(metricObj)
        # print(df)
        pos = np.where(df.index == queryTimestamp)[0][0]
        tbounds, t0, typ = self.calcBounds(df, pos, self.tdelta//2)
        if typ == -1:
            pos1 = df.index[max(0,pos-self.tdelta*2)]
            pos2 = df.index[min(pos+self.tdelta*6,len(df)-1)]
            print(
                f"No ticket drop found in the interval {pos1} and {pos2}. Please try with another timestamp or a higher interval size. Currently generating graphs corresponding to query")
        to_monitor = []
        # typ=-1 # uncomment if want to print all graphs regardless of ticket drop
        if typ != -1:
            anomaly_map, to_monitor = self._calculate_anomalies(
                df, tbounds, to_monitor)
            gpt_str_base = self._create_gpt_str_base(df, t0, typ)
            anomalyObj = self._create_anomaly_obj(
                sorted(anomaly_map.keys(), reverse=True), anomaly_map, tbounds, df)
            gpt_str = ''''''
            headers = ["metric", "anomaly interval mean", "overall mean",
                    "anomaly interval 99th percentile", "overall mean 99th percentile"]
            for idx, head in enumerate(headers):
                if idx == len(headers)-1:
                    gpt_str += head+"\n"
                else:
                    gpt_str += head+","
            for timestamp, objects in anomalyObj.items():
                gpt_str += str(timestamp)+":\n"
                for obj in objects:
                    tmpstr = ""
                    for idx, head in enumerate(headers):
                        if idx == len(headers)-1:
                            tmpstr += (str(obj[head])+"\n")
                        else:
                            tmpstr += (str(obj[head])+",")
                    gpt_str += tmpstr
            gpt_str_base += gpt_str
            # self._save_gpt_str_base(gpt_str_base) # optional: save the formatted gpt input to 'gpt-input.txt' if we want to reuse it
            st=time.time()
            gpt_res = self._openAI_req(gpt_str_base)
            st=time.time()-st
        else:
            gpt_res = ""
            to_monitor = df.columns.tolist()
        vertical = (df.index[t0])
        tickets = ['ss wt concurrentTransactions.write.out',
                   'ss wt concurrentTransactions.read.out']
        for tick in tickets:
            if tick not in to_monitor:
                to_monitor.append(tick)
        st = time.time()
        self.__plot(df.iloc[tbounds[0]:tbounds[-1]],
                    to_monitor, vert_x=vertical, gpt_out=gpt_res) # df.iloc[tbounds[0]:tbounds[-1]] slices the duration of dataframe that we wish to output
        st = time.time()-st
        # print("Time taken to render:",st)

    def parseAll(self):
        """
        Description:
        This method parses all the data from the metrics object and computes certain metrics. This includes performing delta operations on certain metrics, 
        handling new metrics that may appear during the process, and dealing with edge cases where the number of metrics or deltas changes.

        Args:
        None

        Returns: 
        None

        Note:
        This method does not return any value but modifies the internal state of the object by calling the `analytics` method on the processed data.
        """
        def delta(metrList, prevVal=0):
            mylst = [metrList[i] for i in range(len(metrList))]
            for i in range(1, len(metrList)):
                mylst[i] -= metrList[i-1]
            mylst[0] -= prevVal
            return mylst

        def checkCriteria(met):
            if met.startswith("systemMetrics.disks") and met.endswith("io_time_ms"):
                return True
            if met.startswith("replSetGetStatus.members") and (met.endswith("state") or met.endswith("health") or met.endswith("lag")):
                return True
            return False

        data = {}
        iter_keys = iter(self.metricObj)
        # extract the first level of data
        date_string = next(iter_keys)
        data = {}
        metObj = self.metricObj[date_string]
        prevVal = {}
        prevUptime = metObj["serverStatus.uptime"][-1]

        selected_keys = json.load(open('FTDC_metrics.json', 'r'))
        sel_metr_c = selected_keys["to_monitor_c"]
        sel_metr_p = selected_keys["to_monitor_p"]
        locks = selected_keys["locks"]
        for lk in locks["type"]:
            for ops in locks["ops"]:
                for mode in locks["mode"]:
                    new_c_met = "serverStatus.locks."+lk+"."+ops+"."+mode
                    sel_metr_c.append(new_c_met)

        deltactr = len(metObj["serverStatus.start"])
        delta1 = 0
        for met in metObj:
            if met in sel_metr_p:
                data[met] = metObj[met]

        for met in metObj:
            # checkCriteria implements string matching for certain cumulative metrics
            if met in sel_metr_c or checkCriteria(met):
                prevVal[met] = metObj[met][-1]
                data[met] = delta(metObj[met])
        for key in iter_keys:
            metObj = self.metricObj[key]
            try:
                if "serverStatus.uptime" not in metObj or prevUptime > metObj["serverStatus.uptime"][0]:
                    break
            except:
                exit(0)
            sel_metr_c_new = [s for s in metObj.keys() if (
                s in sel_metr_c or checkCriteria(s))]
            sel_metr_p_new = [s for s in metObj.keys() if s in sel_metr_p]

            '''
            handle edge case that a certain metric gets acquired halfway which was not present in the initial list
            eg. serverStatus.locks.Global.acquireWaitTime
            '''
            new_c = [
                item for item in sel_metr_c_new if item not in data]  # metric not in data ever before
            new_p = [item for item in sel_metr_p_new if item not in data]

            for met in new_c:
                # add zeros for those metric who have never been there in the data before
                data[met] = [0 for i in range(deltactr)]
                # print("occurence of new accumulate metric", met)

            for met in new_p:
                data[met] = [0 for i in range(deltactr)]
                # print("occurence of new point metric", met)

            for met in sel_metr_p_new:
                # now fill all the values obtained
                data[met].extend(metObj[met])
            for met in sel_metr_c_new:
                if met in prevVal:
                    previous = prevVal[met]
                else:
                    previous = 0
                prevVal[met] = metObj[met][-1]
                data[met].extend(delta(metObj[met], previous))
            delta1 = len(metObj[sel_metr_p_new[0]])

            # handle the case where unusual number of nmetrics or ndeltas occur,
            # i.e. less metrics are reported compared to previous iteration, so fill with previous values

            for met in data:
                if met not in sel_metr_p_new and not checkCriteria(met) and met not in sel_metr_c:
                    prev_val = data[met][-1]
                    data[met].extend([prev_val] * delta1)
                elif met not in sel_metr_c_new and met not in sel_metr_p:
                    prev_val = data[met][-1]
                    data[met].extend([prev_val] * delta1)

            deltactr += len(metObj["serverStatus.start"])
            if "serverStatus.uptime" in metObj:
                prevUptime = metObj["serverStatus.uptime"][-1]
            else:
                prevUptime = 0
        self.analytics(data, self.queryTimeStamp)
