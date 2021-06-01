# pip install "snowflake-connector-python[secure-local-storage,pandas]"
import snowflake.connector
import pandas as pd
class Downloader():
    def __init__(self, **config):
        """Object to download dataset from snowflake database
        """        
        self.conn = snowflake.connector.connect(
            user = config.get('user'),
            password = config.get('password'),
            account = config.get('account'),
            database = config.get('database'),
            warehouse = config.get('warehouse'),
            schema = config.get('schema')
        )
        self.streamflow = 'streamflow'
        self.station = 'station'

        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')

    def get_by_sql(self, sql):
        """fetch by query language from snowflake database

        Args:
            sql (string): query language

        Returns:
            pd.DataFrame: pandas dataframe of specific dataframe
        """        
        cur = self.conn.cursor().execute(sql)
        return cur.fetch_pandas_all()
    def get_station_infos(self, stations=None):
        """function to get information of all the stations from database

        Returns:
            pd.DataFrame: pandas dataframe of information of all the stations
        """
        if stations == None:
            cur = self.conn.cursor().execute(
                f"SELECT * FROM {self.station}"
            )
        else:
            if not (isinstance(stations, list) or isinstance(stations, tuple)):
                stations = [stations]
            if len(stations) == 1:
                cur = self.conn.cursor().execute(
                    f"SELECT * FROM {self.station} WHERE STAID = {stations[0]}"
                )
            else:
                cur = self.conn.cursor().execute(
                    f"SELECT * FROM {self.station} WHERE STAID in {tuple(stations)}"
                )
        return cur.fetch_pandas_all()
    def get_streamflows(self, stations=None,start=None, end=None):
        """get streamflow information of stations within date range

        Args:
            stations (list, tuple, int, string, optional): the list of stations needs to be fetched, fetch all if None passed. Defaults to None.
            start (string, optional): start date in 'yyyy-mm-dd' format, if None passed fetch downloader's default start date. Defaults to None.
            end (string, optional): end date in 'yyyy-mm-dd' format, if None passed fetch downloader's default end date. Defaults to None.

        Returns:
            pd.DataFrame: pandas dataframe of the quired streamflow in pivot table format with index DATE columns STAID and values STREAMFLOW
        """        
        sql = f'SELECT * from {self.streamflow} WHERE '
        if stations != None:
            if not (isinstance(stations, list) or isinstance(stations, tuple)):
                stations = [stations]
            if len(stations) == 1:
                stations = (stations[0])
                sql += f'STAID = {stations} AND '
            else:
                stations = tuple(stations)
                sql += f'STAID IN {stations} AND '
        if start == None:
            start = self.start_date
        if end == None:
            end = self.end_date
        sql += f"DATE BETWEEN '{start}' AND '{end}'"
        cur = self.conn.cursor().execute(
            sql
        )
        df = cur.fetch_pandas_all()
        if len(df) == 0:
            return pd.DataFrame(index = 'DATE', columns = ['STAID'])
        return df.pivot_table(index='DATE', columns='STAID', values='STREAMFLOW')