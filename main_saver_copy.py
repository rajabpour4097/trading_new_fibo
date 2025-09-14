from fibo_calculate import fibonacci_retracement
import numpy as np
import pandas as pd
from time import sleep
from colorama import init, Fore
from get_legs import get_legs
from get_live_data import get_live_data
from swing import get_swing_points
from utils import BotState
from save_file import log


# راه‌اندازی colorama
init(autoreset=True)

#Initial fib and level
state = BotState()

# initial_swing_search()
state.reset()

cache_data = pd.read_csv("../eurusd_prices_multiip.csv", parse_dates=["timestamp"], index_col="timestamp")
last_data = cache_data.iloc[-1]
start_index = 0
i = 1
print("App started.....")
while True:
    try:
        cache_data = pd.read_csv("../eurusd_prices_multiip.csv", parse_dates=["timestamp"], index_col="timestamp")
        cache_data['status'] = np.where(cache_data['open'] > cache_data['close'], 'bearish', 'bullish')

        if cache_data.iloc[-1].name != last_data.name:
            log(f'                                                                                ')
            log(f'                                                                                ')
            log(f'                                                                                ', color='lightred_ex')
            log(f'Log number {i}:', color='lightred_ex')
            log(f'                                                                                ')
            i += 1
            
            legs = get_legs(cache_data[start_index:])
            log(f'First len legs: {len(legs)}', color='green')
            log(f'                                                                                ')

            if len(legs) > 2:
                legs = legs[-3:]
                swing_type, is_swing = get_swing_points(data=cache_data, legs=legs)

                if is_swing or state.fib_levels is not None:
                    log('1- is_swing or fib_levels is not None', color='blue')
                    log(f"{swing_type} | {cache_data.loc[legs[0]['start']].name} {cache_data.loc[legs[0]['end']].name} "
                        f"{cache_data.loc[legs[1]['start']].name} {cache_data.loc[legs[1]['end']].name} "
                        f"{cache_data.loc[legs[2]['start']].name} {cache_data.loc[legs[2]['end']].name}", color='yellow')

                    log(f'                                                                                ')

                    if not swing_type and state.fib_levels:
                        log(f'not swing_type and fib_levels222', color='yellow')
                        log(f'last_swing_type: {last_swing_type}', color='yellow')
                        
                        if last_swing_type == 'bullish':
                            
                            start_price = legs[1]['end_value']
                            end_price = cache_data.iloc[-1]['high']

                            if state.fib_levels['0.0'] < cache_data.iloc[-1]['high']:
                                log('fib_levels[0.0] < cache_data.iloc[-1][high] 222', color='green')
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                                    
                            elif cache_data.iloc[-1]['low'] <= state.fib_levels['0.705']:
                                if state.last_touched_705_point_up is None:
                                    log('first touch 705 point in 222', color='green')
                                    log(f'                                                                                ')
                                    state.last_touched_705_point_up = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_up['status']:
                                    state.true_position = True
                                    log('true_position = True bullish in 222', color='green')
                            
                            elif cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                log('cache_data.iloc[-1][low] < fib_levels[1.0] 222', color='green')
                                state.reset()
                                legs = legs[-2:]
                                start_index = cache_data.index.tolist().index(legs[0]['start'])
                                
                        elif last_swing_type == 'bearish':
                            
                            end_price = cache_data.iloc[-1]['low']
                            start_price = legs[1]['end_value']
                            
                            if state.fib_levels['0.0'] > cache_data.iloc[-1]['low']:
                                log('fib_levels[0.0] > cache_data.iloc[-1][low] 222', color='green')
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                            
                            elif cache_data.iloc[-1]['high'] >= state.fib_levels['0.705']:
                                if state.last_touched_705_point_down is None:
                                    log('first touch 705 point in 222', color='red')
                                    state.last_touched_705_point_down = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_down['status']:
                                    state.true_position = True
                                    log('true_position = True bearish in 222', color='green')

                            elif cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                log('cache_data.iloc[-1][high] > fib_levels[1.0] 222', color='green')
                                state.reset()
                                legs = legs[-2:]
                                start_index = cache_data.index.tolist().index(legs[0]['start'])

                    elif is_swing and state.fib_levels and last_swing_type == swing_type:
                        
                        log(f'is_swing and fib_levels333', color='yellow')
                        log(f'last_swing_type: {last_swing_type}  now swing_type: {swing_type}', color='yellow')

                        if last_swing_type == 'bullish':
                            
                            start_price = legs[1]['end_value']
                            end_price = cache_data.iloc[-1]['high']

                            if state.fib_levels['0.0'] < cache_data.iloc[-1]['high']:
                                log('fib_levels[0.0] < cache_data.iloc[-1][high] 222', color='green')
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                                    
                            elif cache_data.iloc[-1]['low'] <= state.fib_levels['0.705']:
                                if state.last_touched_705_point_up is None:
                                    log('first touch 705 point in 222', color='green')
                                    log(f'                                                                                ')
                                    state.last_touched_705_point_up = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_up['status']:
                                    state.true_position = True
                                    log('true_position = True bullish in 222', color='green')
                            
                            elif cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                log('cache_data.iloc[-1][low] < fib_levels[1.0] 222', color='green')
                                state.reset()
                                legs = legs[-2:]
                                start_index = cache_data.index.tolist().index(legs[0]['start'])
                                
                        elif last_swing_type == 'bearish':
                            
                            end_price = cache_data.iloc[-1]['low']
                            start_price = legs[1]['end_value']
                            
                            if state.fib_levels['0.0'] > cache_data.iloc[-1]['low']:
                                log('fib_levels[0.0] > cache_data.iloc[-1][low] 222', color='green')
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                            
                            elif cache_data.iloc[-1]['high'] >= state.fib_levels['0.705']:
                                if state.last_touched_705_point_down is None:
                                    log('first touch 705 point in 222', color='red')
                                    state.last_touched_705_point_down = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_down['status']:
                                    state.true_position = True
                                    log('true_position = True bearish in 222', color='green')

                            elif cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                log('cache_data.iloc[-1][high] > fib_levels[1.0] 222', color='green')
                                state.reset()
                                legs = legs[-2:]
                                start_index = cache_data.index.tolist().index(legs[0]['start'])

                    elif swing_type == 'bullish' and state.fib_levels is None:
                        if cache_data.iloc[-1]['close'] >= legs[0]['end_value'] or state.fib_levels:
                            log(f'start Long position(Buy) {cache_data.iloc[-1].name}', color='green')
                            log(f'                                                                                ')
                            
                            start_price = legs[1]['end_value']
                            end_price = cache_data.iloc[-1]['high']
                            if state.fib_levels is None:
                                    state.fib_levels = fibonacci_retracement(end_price, start_price)
                                    fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                    fib_index = cache_data.iloc[-1].name
                                    last_leg1_value = cache_data.index.tolist().index(legs[1]['end'])

                            elif state.fib_levels and state.fib_levels['0.0'] < cache_data.iloc[-1]['high']:
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name

                            if cache_data.iloc[-1]['low'] <= state.fib_levels['0.705']:
                                if state.last_touched_705_point_up is None:
                                    log('first touch 705 point', color='green')
                                    log(f'                                                                                ')
                                    state.last_touched_705_point_up = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_up['status']:
                                    state.true_position = True

                            elif cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                state.reset()
                                legs = legs[-2:]
                                start_index = cache_data.index.tolist().index(legs[0]['start'])

                            log(f'fib_levels: {state.fib_levels}', color='yellow')
                            log(f'fib_index: {fib_index}', color='yellow')


                    elif swing_type == 'bearish' and state.fib_levels is None:
                        if cache_data.iloc[-1]['close'] <= legs[0]['end_value'] or state.fib_levels:
                            log(f'start Short position(Sell) {cache_data.iloc[-1].name}', color='red')
                            log(f'                                                                                ')
                            

                            end_price = cache_data.iloc[-1]['low']
                            start_price = legs[1]['end_value']

                            if state.fib_levels is None:
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                                last_leg1_value = cache_data.index.tolist().index(legs[1]['end'])

                            elif state.fib_levels and state.fib_levels['0.0'] > cache_data.iloc[-1]['low']:
                                state.fib_levels = fibonacci_retracement(end_price, start_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name

                            if cache_data.iloc[-1]['high'] >= state.fib_levels['0.705']:
                                if state.last_touched_705_point_down is None:
                                    log('first touch 705 point', color='red')
                                    state.last_touched_705_point_down = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_down['status']:
                                    state.true_position = True

                            elif cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                state.reset()
                                legs = legs[-2:]
                                start_index = cache_data.index.tolist().index(legs[0]['start'])

                            log(f'fib_levels: {state.fib_levels}', color='yellow')
                            log(f'fib_index: {fib_index}', color='yellow')
                            
                    last_swing_type = swing_type

                    if state.true_position and swing_type == 'bullish':

                        live_data = get_live_data()
                        
                        current_open_point = cache_data.iloc[-1]['close']
                        
                        log(f'Start long position income {cache_data.iloc[-1].name}', color='blue')
                        log(f'current_open_point: {current_open_point}', color='blue')
                        
                        # Initial stop and reward value 
                        if abs(state.fib_levels['0.9'] - current_open_point) * 10000 < 2:
                            stop = state.fib_levels['1.0']
                            log(f'stop = fib_levels[1.0] {stop}', color='red')
                        else:
                            stop = state.fib_levels['0.9']
                            log(f'stop = fib_levels[0.9] {stop}', color='red')

                        stop_distance = abs(current_open_point - stop)
                        reward_end = current_open_point + (stop_distance * 2)

                        log(f'stop = {stop}', color='green')
                        log(f'reward_end = {reward_end}', color='green')
                        log(f'live_data: {live_data}', color='green')

                        # مقدار اولیه high/low
                        high = low = live_data['mid']

                        while high < reward_end and low > stop:
                            
                            log(f'start while bullish mid: {live_data["mid"]}, high: {high}, low: {low}', color='green')
                            
                            live_data = get_live_data()
                            
                            mid = live_data['mid']
                            if mid > high:
                                high = mid
                            if mid < low:
                                low = mid

                            if high >= reward_end:
                                log(f'Reward end toucheddd', color='black')
                            if low <= stop:
                                log(f'Stop point toucheddd', color='black')

                            sleep(0.3)

                        state.reset()
                        legs = []
                        start_index = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                        log(f'End long position, start_index: {start_index}', color='black')

                    if state.true_position and swing_type == 'bearish':
                        
                        live_data = get_live_data()
                        
                        current_open_point = cache_data.iloc[-1]['close']
                        
                        log(f'Start short position income {cache_data.iloc[-1].name}', color='red')
                        log(f'current_open_point: {current_open_point}', color='red')
                        
                        # Initial stop and reward value 
                        if abs(state.fib_levels['0.9'] - current_open_point) * 10000 < 2:
                            stop = state.fib_levels['1.0'] 
                            log(f'stop = fib_levels[1.0] {stop}', color='red')
                        else:
                            stop = state.fib_levels['0.9']
                            log(f'stop = fib_levels[0.9] {stop}', color='red')
                            
                        stop_distance = abs(current_open_point - stop)
                        reward_end = current_open_point - (stop_distance * 2)

                        log(f'stop = {stop}', color='red')
                        log(f'reward_end = {reward_end}', color='red')
                        log(f'live_data: {live_data}', color='red')

                        high = low = live_data['mid']

                        while low > reward_end and high < stop:
                            log(f'start while bearish mid: {live_data["mid"]}, high: {high}, low: {low}', color='red')

                            live_data = get_live_data()
                            
                            mid = live_data['mid']
                            if mid > high:
                                high = mid
                            if mid < low:
                                low = mid

                            if low <= reward_end:
                                log(f'Reward end toucheddd', color='black')
                            if high >= stop:
                                log(f'Stop point toucheddd', color='black')

                            sleep(0.3)

                        state.reset()
                        legs = []
                        start_index = cache_data.index.tolist().index(cache_data.iloc[-1].name)

                        log(f'End short position, start_index: {start_index}', color='black')

                elif is_swing is False and state.fib_levels is None:
                    log('2- is_swing == False and fib_levels is None', color='blue')
                    state.reset()
                    legs = legs[-2:]
                    start_index = cache_data.index.tolist().index(legs[0]['start'])

                    log(f'no swing | Legs: {legs}', color='green')

            log(f'cache_data.iloc[-1].name: {cache_data.iloc[-1].name}', color='lightblue_ex')
            log(f'len(legs): {len(legs)} | start_index: {start_index} | {cache_data.iloc[start_index].name}', color='lightred_ex')

            if len(legs) < 3:
                if len(legs) == 2:
                    log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}, leg1: {legs[1]["start"]}, {legs[1]["end"]}', color='lightcyan_ex')
                if len(legs) == 1:
                    log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}', color='lightcyan_ex')

            log(f'                                                                                ')
            log(f'--------------------------------------------------------------------------------', color='lightred_ex')
            log(f'                                                                                ')

        last_data = cache_data.iloc[-1]
        sleep(0.5)
        

    except Exception as e:
        log(f'                                                                                ')
        log(f"Error: {e}", level='error', color='red')
