"""
This module has a set of functions focused on supporting 
data management in python, it is made to implement the topics seen
in the second module of our bootcamp.



Functions
_____________

    * sql_retrieval: execute a SQL query and returns a Pandas DataFrame with 
                    the data.

@author: Dany
"""
import scipy.stats as stats
import pandas as pd
import psycopg2
import numpy as np

def sql_retrieval(query: str,
                  database: str,
                  user: str = 'postgres',
                  password: str = 'root', 
                  host: str = 'localhost') -> pd.DataFrame:
    """
    Execute the DQL query in PostgreSQL and return a pandas DataFrame with
    the data
    """
    
    conn = psycopg2.connect(host = host,
                            user = user,
                            password = password,
                            database = database)
    
    cur = conn.cursor()
    cur.execute(query)
    
    column_names = [col[0] for col in cur.description]
    data = cur.fetchall()
    
    cur.close()
    conn.close()
    
    df = pd.DataFrame(data, columns = column_names)
    
    return df


    
#x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x    
class Outliers:
    """
    This class focuses on the detection of outliers in a given data set. Use 4 methods
    for this that are listed below:
    
    Z-Score
    -------
        This method uses the following transformation to detect outliers:
        
                                  Z = (x - mean) / std
                                  
        Where mean is the mean of the data and std is its standard deviation, if Z is greater
        or lower than a certain threshold (usually 3), this point is considered an outlier.
        
        This implementation is iterative, once the outliers are detected, the transformation 
        is applied again and outliers are searched again, the process ends when no point 
        exceeds the threshold.
        
        
    Modified Z-Score
    ----------------
        This method is very similar to the Z-Score but uses a different transformation:
                                  
                                  Z = Q(75) * (x - median) / MAD
                                  
        Where Q (75) is the 75th percentile of a standard normal distribution, median is 
        the median of the data, and MAD is the absolute deviation from the median defined as 
        follows:
                                 
                                  MAD = median(|x - median|)
        
        The implementation after transformation is the same as for Z-Score method.
        
    Interquartile Range
    --------------------
        Use the interquartile range (iqr) to detect outliers:
        
                                  iqr = Q3 - Q1
        
        Where Q1 and Q3 are the first and third quartiles respectively. If a value is 
        1.5 * iqr greater than Q3 or 1.5 * iqr less than Q1 it is considered an outlier.
    
    
    Data Trimming
    -------------
        It considers outliers to the k most extreme values or to the k% of the most extreme
        data.
        
        
    ----------
    Attributes
    ----------
    
    multidim → bool
        Read-only property, indicates whether the analysis performed is multidimensional 
        or not.
        
    data → DataFrame
        The data on which the analysis was performed. Always will be a data frame even if 
        the analysis is one-dimensional.
        
    -------
    Methods
    -------
    
    fit
        Runs outlier detection on the provided data. Check the method's documentation for 
        information of the parameters.
    """
    def __init__(self, data = None):
        if isinstance(data, pd.DataFrame):
            self._multidim = len(data.columns) > 1
            self.data      = data
            
        else:
            data = pd.DataFrame(data)
            self._multidim = len(data.columns) > 1
            self.data      = data
            
        
    #Only read Property multi_dim_________________________________________________________________
    @property
    def multidim(self):
        return self._multidim
    
    
    #Data property _______________________________________________________________________________
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if isinstance(data, pd.DataFrame):
            self._data = data      
        else:
            raise ValueError('The Data must be a Pandas DataFrame')
            
            
    #mean distances for multivariable z-score______________________________________________________
    def _mean_distances(self):
        mean = self.data.mean()
        distances = [np.linalg.norm(row - mean) for row in self.data.values]
        
        return pd.Series(distances)
    
    
    #MAD for the Modified Z-score__________________________________________________________________
    def _mad(self, data):
        median = data.median()
        return np.abs(data - median).median()
    
    #z-score method______________________________________________________________________________
    def _z_score(self, data, threshold = 3.0, verbose = False): #only one dimentional method
        data = data.copy()
        outliers = []
        
        while True:
            
            #apply the transform
            mean = data.mean()
            std  = data.std(ddof = 1)
            data_z = (data - mean) / std
            
            to_remove = abs(data_z) > threshold#check for the outliers
            
            if to_remove.values.sum() == 0:
                break   
            else:
                data[to_remove] = np.nan
                outliers.extend(np.where(to_remove)[0])#add current indexes to list
        
        
        if verbose:
            print('{} outliers were found'.format(len(outliers)))
        
        return data, outliers
        
    #Modified z-score method______________________________________________________________________________   
    def _modified_z_score(self, data, threshold = 3.0, verbose = False):#only one dimentional method
        data = data.copy()
        outliers = []
        
        while True:
            #apply the transform
            median = data.median()
            mad  = self._mad(data)
            data_z = (stats.norm.ppf(.75) / mad) * (data - median) 
            
            to_remove = abs(data_z) > threshold#check for the outliers
            
            if to_remove.values.sum() == 0:
                break
            else:
                data[to_remove] = np.nan
                outliers.extend(np.where(to_remove)[0])#add current indexes to list
                
        if verbose:
            print('{} outliers were found'.format(len(outliers)))
        
        return data, np.array(outliers)
        
        
        
    #Data Trimming_________________________________________________________________________________
    def _data_trimming(self, data, verbose = False, k = 5, percentage = True, impute = 'none'):
        data = data.copy()
        
        if percentage:
            data_cut_off = data.quantile(1 - (k / 100))
            data_2_cut = np.where(np.abs(data) > data_cut_off)[0]
            
        else:
            data_sort_idx = np.argsort(np.abs(data))[::-1]
            data_2_cut = data_sort_idx[:k]
        
        #imputation
        if impute.lower() != 'none' and impute in ('median', 'mean', 'extremes'):
            
            if impute == 'median':
                data[data_2_cut] = data.median()
            
            elif impute == 'mean':
                data[data_2_cut] = data.mean()
                
        elif impute.lower() == 'none':
            data[data_2_cut] = np.nan
            
        else:
            raise ValueError('{} it is not recognized as a valid imputation method'.format(impute))
        
        if verbose:
            print('{} outliers were found'.format(len(data_2_cut)))
        
        return data, data_2_cut
        
        
    #interquartile range method___________________________________________________________________
    def _interquartile_range_method(self, data: 'DataFrame',
                                    verbose = False, impute= 'none'):
        data = data.copy()
        
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        
        outliers = (data > upper_limit) | (data < lower_limit)#check for the outliers
        
        #imputation
        if impute.lower() != 'none' and impute in ('median', 'mean', 'extremes'):
            
            if impute == 'median':
                data[outliers] = data.median()
            
            elif impute == 'mean':
                data[outliers] = data.mean()
                
            elif impute == 'extremes':
                data[data > upper_limit] = upper_limit
                data[data < lower_limit] = lower_limit
                
        elif impute.lower() == 'none':
            data[outliers] = np.nan
            
        else:
            raise ValueError('{} it is not recognized as a valid imputation method'.format(impute))
        
        #convert to indices here, before is a boolean mask  
        outliers = np.where(outliers)[0]
        
        if verbose:
            print('{} outliers were found'.format(len(outliers)))
        
        return data, outliers
        
    
    #fit method______________________________________________________________________________________
    def fit(self, data = None,
            how: str = 'iqr',
            impute: str = 'none',
            verbose: bool = False,
            threshold: float = 3.0,
            percentage: bool = True, k: int = 5) -> '(data, outliers)':
        """
        Executes the outlier detection method according to the how parameter,
        returns a copy of the data without outliers and an array with the 
        indices of the outliers.
        
        
        Arguments
        ---------
        
        data = None → DataFrame, Series, Array or list
            The data to detect outliers, it is not necessary if it was provided
            when creating the instance.
            
        how = 'iqr' → str
            The detection method to be used, accepts the following options:
                'iqr'      → For the interquartile range method.
                'zscore'   → For the Z-Score method.
                'mzscore'  → For the Modified Z_Score method.
                'trimming' → For the data trimming method.
                
        impute = 'none' → str
            The way you want to impute outliers, accepts the following options:
                'none'     → It does not impute the outliers, it changes them to NaN's.
                'mean'     → Imputes the value of the mean of the data to the outliers.
                'median'   → Imputes the value of the median of the data to the outliers.
                'extremes' → Only for the interquartile range method. Imputes the value
                             of the closest limit (lower or upper) to the outliers.
           
        verbose = False → bool
            Prints the number of outliers founds.
             
        threshold = 3.0 → float
            The threshold for the Z-Score and Modified Z-Score methods.
        
        percentage = True → bool
            Specifies if you want to follow the criteria of percentage or specific number
            in the data trimming method.
        
        k = 5 → int
            The percentage of the data or the number of outliers considered for the data
            trimming method.
        """
        if isinstance(data, pd.DataFrame):#check the data
            self._multidim = len(data.columns) > 1
            self.data      = data
            
        else:
            data = pd.DataFrame(data)
            self._multidim = len(data.columns) > 1
            self.data      = data
        
        #iqr method_________________________________________________________________________________
        if how == 'iqr':
            data_new = {}.fromkeys(self.data.columns)
            outliers = {}.fromkeys(self.data.columns)

            for k in data_new.keys():
                data_new[k], outliers[k] = self._interquartile_range_method(self.data[k],
                                                                           verbose = verbose,
                                                                           impute = impute)

            return pd.DataFrame(data_new), outliers
        
        #z-score____________________________________________________________________________________
        elif how == 'zscore':
            
            if self.multidim:#multidimentional case
                data = self.data.copy()
                distances = self._mean_distances()
                #keep only the outlier's index
                outliers = self._z_score(distances, threshold = threshold, verbose = verbose)[1]
                
                #imputation
                if impute == 'none':
                    data.iloc[outliers] = np.nan
                    return data, outliers

                elif impute == 'mean':
                    for c in data.columns:
                        data[c][outliers] = data[c].mean()
                        
                    return data, outliers

                elif impute == 'median':
                    for c in data.columns:
                        data[c][outliers] = data[c].median()

                    return data, outliers
                
                else:
                    msn = '{} it is not recognized as a valid imputation method'
                    raise ValueError(msn.format(impute))

            #one dimentional case________________________________________________________________
            else:
                if impute == 'none':
                    return self._z_score(self.data, verbose)

                elif impute == 'mean':
                    data, outliers = self._z_score(self.data, threshold = threshold, verbose = verbose)
                    data.iloc[outliers] = data.mean()[0]
                    return data, outliers

                elif impute == 'median':
                    data, outliers = self._z_score(self.data, threshold = threshold, verbose = verbose)
                    data.iloc[outliers] = data.median()[0]
                    return data, outliers

                else:
                    raise ValueError('{} it is not recognized as a valid imputation method'.format(impute))


        #modified z-score____________________________________________________________________________
        elif how == 'mzscore':
               
            #multidimentional case
            if self.multidim:
                data = self.data.copy()
                distances = self._mean_distances()
                #keep only the outlier's index
                outliers = self._modified_z_score(distances, threshold = threshold, 
                                                  verbose = verbose)[1]
                
                #imputation
                if impute == 'none':
                    data.iloc[outliers] = np.nan
                    return data, outliers

                elif impute == 'mean':
                    for c in data.columns:
                        data[c][outliers] = data[c].mean()

                    return data, outliers

                elif impute == 'median':
                    for c in data.columns:
                        data[c][outliers] = data[c].median()

                    return data, outliers

                else:
                    msn = '{} it is not recognized as a valid imputation method'
                    raise ValueError(msn.format(impute))

            #one dimentional case________________________________________________________________
            else:
                if impute == 'none':
                    return self._modified_z_score(self.data, verbose)

                elif impute == 'mean':
                    data, outliers = self._modified_z_score(self.data, threshold = threshold,
                                         verbose = verbose)
                    data.iloc[outliers] = data.mean()[0]
                    return data, outliers

                elif impute == 'median':
                    data, outliers = self._modified_z_score(self.data, threshold = threshold,
                                         verbose = verbose)
                    data.iloc[outliers] = data.median()[0]
                    return data, outliers
                
                else:
                    msn = '{} it is not recognized as a valid imputation method'
                    raise ValueError(msn.format(impute))

        #data trimming________________________________________________________________________________
        elif how == 'trimming':
            data_new = {}.fromkeys(self.data.columns)
            outliers = {}.fromkeys(self.data.columns)

            for c in data_new.keys():
                data_new[c], outliers[c] = self._data_trimming(self.data[c], k = k, verbose = verbose,
                                                               percentage = percentage,
                                                               impute = impute)
            return pd.DataFrame(data_new), outliers
    
