################################################################################
# Compute directional change as per definitions in Appendix A from
# Detecting Regime Change in Computational Finance by Jun Chen, Edward P K Tsang
# Rohan Prasad
##############################################################################
import pandas as pd


def get_pct_change(start, end):
    return (end - start) / start


def get_dc_data_test(prices: pd.Series, theta: float) -> list[tuple]:
    """

    :param prices: prices
    :param theta: threshold
    :return: Returns a list of tuples. Each tuple is of the form
             (Directional Change Confirmation timestamp, Directional Change Confirmation price,
             Downturn/Upturn time, Downturn/Upturn price)
    """

    last_high = last_low = prices[0]
    last_low_time = last_high_time = prices.index[0]
    is_upward_run = is_downward_run = is_downward_overshoot = is_upward_overshoot = False
    ret_val = []

    for timestamp, current_price in prices[1:].items():
        if get_pct_change(last_high, current_price) <= -theta:
            is_downward_run = True
            is_upward_run = False
            is_upward_overshoot = False
            if is_downward_overshoot:
                last_high = current_price
                last_high_time = timestamp
            else:
                # I have a confirmation point
                print('Starting Downward Run. Current Price: {:.4f} Last Low: {:.4f} Last High: {:.4f} '.format(
                    current_price, last_low, last_high))
                ret_val.append((timestamp, current_price, last_high_time, last_high))
                is_downward_overshoot = True
        elif get_pct_change(last_low, current_price) >= theta:
            is_upward_run = True
            is_downward_run = False
            is_downward_overshoot = False
            if is_upward_overshoot:
                last_low = current_price
                last_low_time = timestamp
            else:
                ret_val.append((timestamp, current_price, last_low_time, last_low))
                is_upward_overshoot = True
        if last_low > current_price:
            last_low = current_price
            last_low_time = timestamp
        if last_high < current_price:
            last_high = current_price
            last_high_time = timestamp
    return ret_val


if __name__ == '__main__':
    print('Please import this file as a module.')  # %%
