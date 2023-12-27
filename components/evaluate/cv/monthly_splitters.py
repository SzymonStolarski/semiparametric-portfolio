from abc import ABC, abstractmethod

import pandas as pd


class BaseMonthlySplitter(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def split_generator(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def _get_splits(self):
        pass


class ExpandingMonthlySplitter(BaseMonthlySplitter):

    def __init__(
            self,
            initial_window: int,
            step_length: int,
            test_horizon: int) -> None:

        self.initial_window = initial_window
        self.step_length = step_length
        self.test_horizon = test_horizon


    def split_generator(self, X: pd.DataFrame):
        splits = self._get_splits(X)

        return splits.iterrows()
    
    def _get_splits(self, df: pd.DataFrame):

        dateranges_m_start = [
            date for date in
            pd.date_range(df.index[0], df.index[-1], freq='MS')]
        dateranges_m_end = [
            date + pd.tseries.offsets.MonthEnd(0)
            for date in pd.date_range(df.index[0], df.index[-1], freq='MS')]

        # Need to delete -1 from len as it is 1 larger than idx range
        # Also, train end needs to be shorter than the test horizon start
        train_end_idx = [
            i+1 for i
            in range(self.initial_window,
                     len(dateranges_m_start)-1-self.test_horizon, self.step_length)]
        train_end_dates = [dateranges_m_end[idx] for idx in train_end_idx]

        test_start_idx = [i+1 for i in train_end_idx]
        test_start_dates = [dateranges_m_start[idx] for idx in test_start_idx]

        # Cases for the test horizon
        # If it's one, then start of the test period is beggining of the month
        # and ending at the same month
        if self.test_horizon == 1:
            test_end_idx = test_start_idx
            test_end_dates = [dateranges_m_end[idx] for idx in test_end_idx]
        # If it's larger than one, then it's beggining at the test start month
        # and ending at the end of a particular month in n interval
        elif self.test_horizon > 1:
            t_horizon = self.test_horizon-1
            test_end_idx = [i+t_horizon for i in test_start_idx]
            test_end_dates = [dateranges_m_end[idx] for idx in test_end_idx]

        df_split_dates = pd.DataFrame({
            'train_start': [df.index[0] for _ in range(len(train_end_idx))],
            'train_end': train_end_dates,
            'test_start': test_start_dates,
            'test_end': test_end_dates
        })

        return df_split_dates


class SlidingMonthlySplitter(BaseMonthlySplitter):

    def __init__(
            self, initial_window: int,
            step_length: int,
            test_horizon: int,
            window_length: int) -> None:

        self.initial_window = initial_window
        self.step_length = step_length
        self.test_horizon = test_horizon
        self.window_length = window_length


    def split_generator(self, X: pd.DataFrame):
        splits = self._get_splits(X)

        return splits.iterrows()
    
    def _get_splits(self, df: pd.DataFrame):

        dateranges_m_start = [
            date for date in
            pd.date_range(df.index[0], df.index[-1], freq='MS')]
        dateranges_m_end = [
            date + pd.tseries.offsets.MonthEnd(0)
            for date in pd.date_range(df.index[0], df.index[-1], freq='MS')]

        # Need to delete -1 from len as it is 1 larger than idx range
        # Also, train end needs to be shorter than the test horizon start
        train_end_idx = [
            i+1 for i
            in range(self.initial_window,
                     len(dateranges_m_start)-1-self.test_horizon, self.step_length)]
        train_end_dates = [dateranges_m_end[idx] for idx in train_end_idx]

        train_start_idx = [i+1-self.window_length for i in train_end_idx]
        train_start_dates = [dateranges_m_start[idx] for idx in train_start_idx]

        test_start_idx = [i+1 for i in train_end_idx]
        test_start_dates = [dateranges_m_start[idx] for idx in test_start_idx]

        # Cases for the test horizon
        # If it's one, then start of the test period is beggining of the month
        # and ending at the same month
        if self.test_horizon == 1:
            test_end_idx = test_start_idx
            test_end_dates = [dateranges_m_end[idx] for idx in test_end_idx]
        # If it's larger than one, then it's beggining at the test start month
        # and ending at the end of a particular month in n interval
        elif self.test_horizon > 1:
            t_horizon = self.test_horizon-1
            test_end_idx = [i+t_horizon for i in test_start_idx]
            test_end_dates = [dateranges_m_end[idx] for idx in test_end_idx]

        df_split_dates = pd.DataFrame({
            'train_start': train_start_dates,
            'train_end': train_end_dates,
            'test_start': test_start_dates,
            'test_end': test_end_dates
        })

        return df_split_dates