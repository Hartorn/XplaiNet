import numpy as np
from sklearn.preprocessing import LabelEncoder

UNKNOWN_VALUE = ["Unkn0wnV@lue"]


class SafeLabelEncoder(LabelEncoder):
    """
    Safe label encoder, encoding every unknown value as Unkn0wnV@lue.
    """

    def fit(self, y):
        """
        Fit the label encoder, by casting the numpy array as a string, then adding the code for unknown.
        
        Parameters
        ----------
        y : numpy array
            the values to fit
        
        Returns
        -------
        SafeLabelEncoder
            itself, fitted
        """
        return super().fit(np.concatenate((y.astype("str"), UNKNOWN_VALUE)))

    def fit_transform(self, y):
        """
        Fit the encoder, then transform the input data and returns it.
        
        Parameters
        ----------
        y : numpy array
            the values to fit
        
        Returns
        -------
        numpy array
            the encoded data
        """
        self.fit(y)
        return super().transform(y)

    def transform(self, y):
        """
        Transform the input data and returns it.
        
        Parameters
        ----------
        y : numpy array
            the values to fit
        
        Returns
        -------
        numpy array
            the encoded data
        """
        return super().transform(
            np.where(
                np.isin(y.astype("str"), self.classes_), y.astype("str"), UNKNOWN_VALUE
            )
        )

