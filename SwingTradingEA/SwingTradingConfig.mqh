//+------------------------------------------------------------------+
//|                                            SwingTradingConfig.mqh |
//|                         Configuration file for Swing Trading EA  |
//|                                                                  |
//+------------------------------------------------------------------+

#ifndef SWING_TRADING_CONFIG_MQH
#define SWING_TRADING_CONFIG_MQH

//--- Trading Configuration (equivalent to TRADING_CONFIG in Python)
#define THRESHOLD_DEFAULT           6        // Default leg threshold (pips)
#define FIB_705_LEVEL              0.705     // Fibonacci 70.5% level
#define FIB_90_LEVEL               0.9       // Fibonacci 90% level  
#define WINDOW_SIZE_DEFAULT        100       // Default data window size
#define MIN_SWING_SIZE_DEFAULT     4         // Minimum swing size
#define ENTRY_TOLERANCE_DEFAULT    2.0       // Entry tolerance (pips)
#define LOOKBACK_PERIOD_DEFAULT    20        // Lookback period for analysis

//--- MT5 Configuration (equivalent to MT5_CONFIG in Python)
#define SYMBOL_DEFAULT             "EURUSD"  // Default trading symbol
#define LOT_SIZE_DEFAULT           0.01      // Default lot size
#define WIN_RATIO_DEFAULT          1.2       // Default win ratio (TP/SL)
#define MAGIC_NUMBER_DEFAULT       234000    // Default magic number
#define DEVIATION_DEFAULT          20        // Price deviation tolerance
#define MAX_SPREAD_DEFAULT         3.0       // Maximum spread (pips)
#define MIN_BALANCE_DEFAULT        100       // Minimum account balance
#define MAX_DAILY_TRADES_DEFAULT   10        // Maximum trades per day

//--- Trading Hours (Iran Time Zone GMT+3:30)
#define TRADING_START_HOUR         9         // Trading start hour (Iran time)
#define TRADING_START_MINUTE       0         // Trading start minute
#define TRADING_END_HOUR           21        // Trading end hour (Iran time)  
#define TRADING_END_MINUTE         0         // Trading end minute

//--- Different trading sessions in Iran time
struct TradingSession {
    int startHour;
    int startMinute;
    int endHour;
    int endMinute;
    string name;
};

//--- Sydney session (05:30 - 14:30 Iran time)
const TradingSession SYDNEY_SESSION = {5, 30, 14, 30, "Sydney"};

//--- Tokyo session (07:30 - 16:30 Iran time)
const TradingSession TOKYO_SESSION = {7, 30, 16, 30, "Tokyo"};

//--- London session (12:30 - 21:30 Iran time)
const TradingSession LONDON_SESSION = {12, 30, 21, 30, "London"};

//--- New York session (17:30 - 02:30 Iran time)
const TradingSession NEWYORK_SESSION = {17, 30, 2, 30, "NewYork"};

//--- London-NY overlap (17:30 - 21:30 Iran time) - Best trading time
const TradingSession OVERLAP_LONDON_NY = {17, 30, 21, 30, "London-NY Overlap"};

//--- Iran active hours (09:00 - 21:00)
const TradingSession IRAN_ACTIVE = {9, 0, 21, 0, "Iran Active"};

//--- 24 hours trading
const TradingSession FULL_TIME = {0, 0, 23, 59, "24 Hours"};

//--- Log Configuration
#define LOG_LEVEL_DEBUG            0
#define LOG_LEVEL_INFO             1
#define LOG_LEVEL_WARNING          2
#define LOG_LEVEL_ERROR            3

#define LOG_LEVEL_DEFAULT          LOG_LEVEL_INFO
#define LOG_TO_FILE_DEFAULT        true
#define MAX_LOG_SIZE_MB            10

//--- Risk Management
#define MAX_RISK_PERCENT           2.0       // Maximum risk per trade (% of balance)
#define MAX_DRAWDOWN_PERCENT       10.0      // Maximum allowed drawdown (%)
#define MIN_FREE_MARGIN_PERCENT    50.0      // Minimum free margin required (%)

//--- Market Condition Filters
#define MIN_VOLATILITY             5.0       // Minimum volatility (pips)
#define MAX_VOLATILITY             50.0      // Maximum volatility (pips)
#define NEWS_FILTER_MINUTES        30        // Minutes to avoid trading around news

//--- Position Management
#define PARTIAL_CLOSE_PERCENT      50        // Partial close percentage at certain profit
#define TRAILING_STOP_PIPS         10        // Trailing stop distance (pips)
#define BREAK_EVEN_PIPS            10        // Move SL to break-even after X pips profit

//--- Fibonacci Levels (complete set)
struct FibonacciLevels {
    double level_0;      // 0.0
    double level_236;    // 23.6%
    double level_382;    // 38.2%
    double level_5;      // 50.0%
    double level_618;    // 61.8%
    double level_705;    // 70.5% (custom level used in strategy)
    double level_786;    // 78.6%
    double level_9;      // 90.0% (custom level used in strategy)
    double level_100;    // 100.0%
};

//--- Error Codes
#define ERROR_NONE                 0
#define ERROR_INVALID_PARAMETERS   1
#define ERROR_INSUFFICIENT_FUNDS   2
#define ERROR_TRADING_DISABLED     3
#define ERROR_INVALID_STOPS        4
#define ERROR_MARKET_CLOSED        5

//--- Helper macros
#define POINTS_TO_PIPS(points)     ((points) / 10.0)
#define PIPS_TO_POINTS(pips)       ((pips) * 10.0)
#define IS_BULLISH_CANDLE(rates, i) ((rates)[(i)].close >= (rates)[(i)].open)
#define IS_BEARISH_CANDLE(rates, i) ((rates)[(i)].close < (rates)[(i)].open)

//--- Function to check if current time is within trading session
bool IsWithinTradingSession(const TradingSession &session) {
    MqlDateTime currentTime;
    TimeToStruct(TimeGMT() + 12600, currentTime); // Convert to Iran time (GMT+3:30)
    
    int currentMinutes = currentTime.hour * 60 + currentTime.min;
    int startMinutes = session.startHour * 60 + session.startMinute;
    int endMinutes = session.endHour * 60 + session.endMinute;
    
    // Handle overnight sessions (like NY session)
    if(startMinutes > endMinutes) {
        return (currentMinutes >= startMinutes || currentMinutes <= endMinutes);
    }
    else {
        return (currentMinutes >= startMinutes && currentMinutes <= endMinutes);
    }
}

//--- Function to get current Iran time as string
string GetIranTimeString() {
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime); // GMT+3:30 for Iran
    return StringFormat("%04d.%02d.%02d %02d:%02d:%02d", 
                       iranTime.year, iranTime.mon, iranTime.day,
                       iranTime.hour, iranTime.min, iranTime.sec);
}

//--- Function to check if it's weekend in Iran
bool IsIranWeekend() {
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime);
    
    // In Iran: Thursday = 4, Friday = 5 (weekend)
    // But forex market: Friday evening to Sunday evening is closed
    return (iranTime.day_of_week == 5 || iranTime.day_of_week == 6);
}

#endif // SWING_TRADING_CONFIG_MQH
//+------------------------------------------------------------------+
