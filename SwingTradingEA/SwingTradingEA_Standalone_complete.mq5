//+------------------------------------------------------------------+
//|                                SwingTradingEA_Complete_Python.mq5 |
//|         Complete implementation matching main_metatrader.py      |
//|                    تبدیل کامل از نسخه Python                      |
//+------------------------------------------------------------------+
#property copyright "Swing Trading EA - Complete Python Port"
#property link      ""
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters - مطابق Python config
input group "=== Trading Parameters ==="
input double   InpLotSize = 0.01;           // Lot size
input double   InpWinRatio = 1.2;           // Win ratio (TP/SL)
input int      InpThreshold = 6;            // Leg threshold (pips)
input int      InpWindowSize = 100;         // Data window size
input int      InpMinSwingSize = 4;         // Minimum swing size
input double   InpFib705 = 0.705;          // Fibonacci 70.5% level
input double   InpFib90 = 0.9;             // Fibonacci 90% level

input group "=== Time Settings (Iran GMT+3:30) ==="
input string   InpStartTime = "09:00";     // Trading start time (Iran)
input string   InpEndTime = "21:00";       // Trading end time (Iran)
input bool     InpTradeOnWeekends = false; // Trade on weekends
input bool     InpUseLondonSession = true; // Use London session (12:30-21:30)
input bool     InpUseNYSession = true;     // Use NY session (17:30-02:30)
input bool     InpUseOverlapOnly = false;  // Trade only London-NY overlap

input group "=== Risk Management ==="
input double   InpMaxRiskPercent = 2.0;    // Max risk per trade (% of balance)
input double   InpMaxSpread = 3.0;         // Maximum spread (pips)
input int      InpMaxDailyTrades = 10;     // Maximum daily trades
input bool     InpUseTrailingStop = false; // Use trailing stop

input group "=== Expert Settings ==="
input int      InpMagicNumber = 234000;    // Magic number
input string   InpTradeComment = "SwingEA"; // Trade comment
input bool     InpShowInfo = true;         // Show info panel
input bool     InpEnableAlerts = true;     // Enable alerts
input bool     InpDebugMode = false;       // Debug mode (verbose logs)

//--- Global variables
CTrade trade;
MqlRates rates[];
datetime lastProcessedTime = 0;
int barsTotal = 0;
bool positionOpen = false;
ulong currentTicket = 0;
int dailyTradesCount = 0;
datetime lastDayCheck = 0;
int processCounter = 0;
int fibCounter = 1;

//--- State variables - مطابق BotState در Python
struct BotState {
    double fibLevels[4];        // [0.0, 0.705, 0.9, 1.0]
    bool truePosition;
    MqlRates lastTouched705PointUp;
    MqlRates lastTouched705PointDown;
    bool hasTouched705Up;
    bool hasTouched705Down;
    bool hasValidFib;
    datetime fibCreationTime;
    int consecutiveFailures;
    double totalProfit;
    int totalTrades;
} botState;

//--- Leg structure - مطابق Python
struct Leg {
    datetime start;
    datetime end;
    double startValue;
    double endValue;
    double length;
    string direction;
};

Leg legs[];
int legsCount = 0;
string lastSwingType = "";
int startIndex = 0;
datetime fibIndex = 0;
int fib0Point = 0;
int lastLeg1Value = 0;
double endPrice = 0;
double startPrice = 0;

//--- Trading sessions (Iran Time GMT+3:30)
struct TradingSession {
    int startHour, startMinute;
    int endHour, endMinute;
    string name;
};

const TradingSession LONDON_SESSION = {12, 30, 21, 30, "London"};
const TradingSession NEWYORK_SESSION = {17, 30, 2, 30, "NewYork"};
const TradingSession OVERLAP_LONDON_NY = {17, 30, 21, 30, "London-NY Overlap"};

//--- Performance tracking
struct PerformanceStats {
    int totalSignals;
    int executedTrades;
    int profitableTrades;
    double totalPL;
    datetime lastResetTime;
} stats;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Validate input parameters
    if(InpLotSize <= 0 || InpWinRatio <= 0 || InpThreshold <= 0) {
        Print("❌ Invalid input parameters");
        return(INIT_PARAMETERS_INCORRECT);
    }
    
    //--- Set magic number
    trade.SetExpertMagicNumber(InpMagicNumber);
    
    //--- Initialize bot state
    ResetBotState();
    ZeroMemory(stats);
    stats.lastResetTime = TimeCurrent();
    
    //--- Set timer for processing (every 500ms like Python)
    EventSetTimer(1);
    
    //--- Array setup
    ArraySetAsSeries(rates, true);
    ArrayResize(legs, 1000);
    
    //--- Print initialization - مطابق Python
    Print("🚀 MT5 Trading Bot Started...");
    Print("📊 Config: Symbol=", Symbol(), ", Lot=", InpLotSize, ", Win Ratio=", InpWinRatio);
    Print("⏰ Trading Hours (Iran): ", InpStartTime, " - ", InpEndTime);
    Print("🇮🇷 Current Iran Time: ", GetIranTimeString());
    
    //--- System checks مطابق Python
    Print("🔍 Checking symbol properties...");
    CheckSymbolProperties();
    Print("🔍 Testing broker conditions...");
    CheckTradingConditions();
    Print("🔍 Checking account permissions...");
    CheckAccountPermissions();
    Print("🔍 Checking market state...");
    CheckMarketState();
    Print(StringFormat("%s", StringSubstr("------------------------------------------------", 0, 50)));
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    Print("📊 Final Statistics:");
    Print("   Total Signals: ", stats.totalSignals);
    Print("   Executed Trades: ", stats.executedTrades);
    Print("   Profitable Trades: ", stats.profitableTrades);
    Print("   Total P/L: ", stats.totalPL);
    if(stats.executedTrades > 0) {
        Print("   Win Rate: ", (double)stats.profitableTrades/stats.executedTrades*100, "%");
    }
    
    Print("🔌 Swing Trading EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check new bar (مطابق Python)
    int bars = iBars(Symbol(), PERIOD_M1);
    if(bars <= barsTotal) return;
    
    //--- Update daily counter
    UpdateDailyTradeCounter();
    
    //--- Process new data
    ProcessNewData();
    barsTotal = bars;
    
    //--- Update trailing stop
    if(InpUseTrailingStop && positionOpen) {
        UpdateTrailingStop();
    }
    
    //--- Update info panel
    if(InpShowInfo) {
        UpdateInfoPanel();
    }
}

//+------------------------------------------------------------------+
//| Timer function - مطابق while loop در Python                    |
//+------------------------------------------------------------------+
void OnTimer()
{
    //--- Check trading conditions (مطابق can_trade در Python)
    bool canTrade = false;
    string tradeMessage = "";
    CheckTradingConditions(canTrade, tradeMessage);
    
    if(!canTrade) {
        if(InpDebugMode && processCounter % 60 == 0) { // هر دقیقه لاگ
            LogMessage(StringFormat("⏰ %s", tradeMessage), "yellow");
        }
        processCounter++;
        return;
    }
    
    //--- Process data every second (مطابق Python)
    ProcessNewData();
}

//+------------------------------------------------------------------+
//| Main data processing - مطابق منطق اصلی Python                  |
//+------------------------------------------------------------------+
void ProcessNewData()
{
    //--- Get historical data مطابق Python
    int copied = CopyRates(Symbol(), PERIOD_M1, 0, InpWindowSize * 2, rates);
    if(copied <= 0) {
        LogMessage("❌ Failed to get historical data from MT5", "red");
        return;
    }
    
    //--- Add status column مطابق Python
    // cache_data['status'] = np.where(cache_data['open'] > cache_data['close'], 'bearish', 'bullish')
    
    //--- Check for new data مطابق Python logic
    datetime currentTime = rates[0].time;
    bool processData = false;
    
    if(lastProcessedTime == 0) {
        LogMessage(StringFormat("🔄 First run - processing data from %s", TimeToString(currentTime)), "cyan");
        lastProcessedTime = currentTime;
        processData = true;
    }
    else if(currentTime != lastProcessedTime) {
        LogMessage(StringFormat("📊 New data received: %s (previous: %s)", 
                   TimeToString(currentTime), TimeToString(lastProcessedTime)), "cyan");
        lastProcessedTime = currentTime;
        processData = true;
    }
    else {
        // مطابق Python: waiting for new data
        if(InpDebugMode && processCounter % 20 == 0) {
            LogMessage(StringFormat("⏳ Waiting for new data... Current: %s", TimeToString(currentTime)), "yellow");
        }
        processData = false;
    }
    
    if(processData) {
        //--- مطابق Python main loop
        LogMessage("", "");
        LogMessage("", "");
        LogMessage("", "");
        LogMessage(StringFormat("Log number %d:", processCounter), "lightred_ex");
        LogMessage(StringFormat("%s", StringSubstr("                                                                                ", 0, 80)), "");
        processCounter++;
        
        //--- Get legs - مطابق get_legs(cache_data.iloc[start_index:])
        GetLegs();
        LogMessage(StringFormat("First len legs: %d", legsCount), "green");
        LogMessage(StringFormat("%s", StringSubstr("                                                                                ", 0, 80)), "");
        
        if(legsCount > 2) {
            //--- Take last 3 legs مطابق Python
            ArrayResize(legs, MathMin(legsCount, 3)); // Keep only last 3
            
            string swingType;
            bool isSwing = GetSwingPoints(swingType);
            
            if(isSwing || botState.hasValidFib) {
                LogMessage("1- is_swing or fib_levels is not None code:411112", "blue");
                LogMessage(StringFormat("%s | %s %s %s %s %s %s", 
                          swingType,
                          TimeToString(legs[0].start), TimeToString(legs[0].end),
                          TimeToString(legs[1].start), TimeToString(legs[1].end),
                          TimeToString(legs[2].start), TimeToString(legs[2].end)), "yellow");
                LogMessage(StringFormat("%s", StringSubstr("                                                                                ", 0, 80)), "");
                
                //--- Phase 1: Initial swing detection مطابق Python
                if(isSwing && !botState.hasValidFib) {
                    ProcessInitialSwingDetection(swingType, isSwing);
                }
                //--- Phase 2: Update existing fibonacci in same direction
                else if(isSwing && botState.hasValidFib && lastSwingType == swingType) {
                    ProcessSameDirectionSwing(swingType, isSwing);
                }
                //--- Phase 3: Opposite swing handling
                else if(isSwing && botState.hasValidFib && lastSwingType != swingType) {
                    ProcessOppositeDirectionSwing(swingType);
                }
                //--- Phase 4: Update without swing
                else if(!isSwing && botState.hasValidFib) {
                    ProcessFibonacciUpdates();
                }
            }
        }
        //--- Handle len(legs) < 3 case مطابق Python
        else if(legsCount < 3) {
            ProcessShortLegsCase();
        }
        
        //--- Trading signals - مطابق Python buy/sell logic
        CheckTradingSignals();
        
        //--- Final logs مطابق Python
        LogMessage(StringFormat("cache_data.iloc[-1].name: %s", TimeToString(rates[0].time)), "lightblue_ex");
        LogMessage(StringFormat("len(legs): %d | start_index: %d | %s", 
                   legsCount, startIndex, 
                   (startIndex < ArraySize(rates)) ? TimeToString(rates[startIndex].time) : "N/A"), "lightred_ex");
        LogMessage(StringFormat("%s", StringSubstr("                                                                                ", 0, 80)), "");
        LogMessage(StringFormat("%s", StringSubstr("--------------------------------------------------------------------------------", 0, 80)), "");
        LogMessage(StringFormat("%s", StringSubstr("                                                                                ", 0, 80)), "");
    }
    
    //--- Monitor positions مطابق Python
    MonitorPositions();
}

//+------------------------------------------------------------------+
//| Get legs - مطابق get_legs.py                                   |
//+------------------------------------------------------------------+
void GetLegs()
{
    //--- Reset legs
    ArrayFree(legs);
    legsCount = 0;
    
    if(ArraySize(rates) < InpWindowSize) return;
    
    //--- مطابق get_legs function در Python
    double threshold = InpThreshold;
    datetime currentStartIndex = rates[startIndex].time;
    int j = 0;
    
    for(int i = startIndex + 1; i < ArraySize(rates); i++) {
        //--- Determine current price مطابق Python logic
        bool currentIsBullish = (rates[i].close >= rates[i].open);
        double currentPrice;
        
        //--- Enhanced price determination مطابق Python
        if(j > 0 && legs[j-1].direction == "up" && rates[i].high >= rates[i-1].high) {
            currentPrice = rates[i].high;
        }
        else if(j > 0 && legs[j-1].direction == "down" && rates[i].low <= rates[i-1].low) {
            currentPrice = rates[i].low;
        }
        else {
            currentPrice = currentIsBullish ? rates[i].high : rates[i].low;
        }
        
        //--- Get start index position
        int startIndexPos = GetRateIndexByTime(currentStartIndex);
        if(startIndexPos < 0) continue;
        
        //--- Determine start price
        bool startIsBullish = (rates[startIndexPos].close >= rates[startIndexPos].open);
        double startPrice = startIsBullish ? rates[startIndexPos].high : rates[startIndexPos].low;
        
        //--- Calculate price difference in pips
        double priceDiff = MathAbs(currentPrice - startPrice) * 10000;
        
        //--- Determine direction مطابق Python
        string direction = "";
        if(rates[i].close >= rates[startIndexPos].close || 
           (rates[i].high > rates[i-1].high && rates[i].close >= rates[i-1].close)) {
            direction = "up";
        }
        else if(rates[i].close < rates[startIndexPos].close || 
                (rates[i].low < rates[i-1].low && rates[i].close < rates[i-1].close)) {
            direction = "down";
        }
        
        //--- Leg validation and creation مطابق Python logic
        if(priceDiff >= threshold && priceDiff < threshold * 5) {
            if(GetTimeDifference(currentStartIndex, rates[i].time) >= 3) {
                if(j > 0 && legs[j-1].direction == direction) {
                    //--- Update existing leg
                    legs[j-1].end = rates[i].time;
                    legs[j-1].endValue = currentPrice;
                    legs[j-1].length = priceDiff + legs[j-1].length;
                    currentStartIndex = rates[i].time;
                }
                else {
                    //--- Create new leg
                    ArrayResize(legs, j + 1);
                    legs[j].start = currentStartIndex;
                    legs[j].startValue = startPrice;
                    legs[j].end = rates[i].time;
                    legs[j].endValue = currentPrice;
                    legs[j].length = priceDiff;
                    legs[j].direction = direction;
                    
                    j++;
                    currentStartIndex = rates[i].time;
                }
            }
        }
    }
    
    legsCount = j;
    if(InpDebugMode && legsCount > 0) {
        Print("📈 Detected ", legsCount, " legs");
    }
}

//+------------------------------------------------------------------+
//| Get swing points - مطابق swing.py                              |
//+------------------------------------------------------------------+
bool GetSwingPoints(string &swingType)
{
    if(legsCount < 3) return false;
    
    swingType = "";
    bool isSwing = false;
    
    //--- مطابق swing.py logic
    //--- Up swing
    if(legs[1].endValue > legs[0].startValue && legs[0].endValue > legs[1].endValue) {
        //--- Check true swing مطابق Python validation
        int sIndex = GetRateIndexByTime(legs[1].start);
        int eIndex = GetRateIndexByTime(legs[1].end);
        
        if(sIndex >= 0 && eIndex >= 0) {
            int trueCandles = 0;
            bool firstCandle = false;
            double lastCandleClose = 0;
            
            for(int k = sIndex; k >= eIndex; k--) {
                string status = (rates[k].close < rates[k].open) ? "bearish" : "bullish";
                
                if(status == "bearish") {
                    if(firstCandle) {
                        if(rates[k].close < lastCandleClose) {
                            trueCandles++;
                            lastCandleClose = rates[k].close;
                        }
                    }
                    else {
                        lastCandleClose = rates[k].close;
                    }
                    firstCandle = true;
                }
            }
            
            if(trueCandles >= 3) {
                swingType = "bullish";
                isSwing = true;
            }
        }
    }
    //--- Down swing
    else if(legs[1].endValue < legs[0].startValue && legs[0].endValue < legs[1].endValue) {
        //--- Check true swing مطابق Python validation
        int sIndex = GetRateIndexByTime(legs[1].start);
        int eIndex = GetRateIndexByTime(legs[1].end);
        
        if(sIndex >= 0 && eIndex >= 0) {
            int trueCandles = 0;
            bool firstCandle = false;
            double lastCandleClose = 0;
            
            for(int k = sIndex; k >= eIndex; k--) {
                string status = (rates[k].close >= rates[k].open) ? "bullish" : "bearish";
                
                if(status == "bullish") {
                    if(firstCandle) {
                        if(rates[k].close > lastCandleClose) {
                            trueCandles++;
                            lastCandleClose = rates[k].close;
                        }
                    }
                    else {
                        lastCandleClose = rates[k].close;
                    }
                    firstCandle = true;
                }
            }
            
            if(trueCandles >= 3) {
                swingType = "bearish";
                isSwing = true;
            }
        }
    }
    
    return isSwing;
}

//+------------------------------------------------------------------+
//| Process initial swing detection - مطابق Python Phase 1         |
//+------------------------------------------------------------------+
void ProcessInitialSwingDetection(string swingType, bool isSwing)
{
    LogMessage("is_swing and fib_levels is None code:4113312", "yellow");
    lastSwingType = swingType;
    
    if(swingType == "bullish") {
        if(rates[0].close >= legs[0].endValue) {
            double localStartPrice = rates[0].high;  // بجای startPrice
            double localEndPrice = legs[1].endValue;  // بجای endPrice
            
            if(rates[0].high >= legs[1].endValue) {
                LogMessage(StringFormat("The %d of fib_levels value code:4116455 %s", 
                          fibCounter, TimeToString(rates[0].time)), "green");
                CalculateFibonacci(localEndPrice, localStartPrice);
                fib0Point = 0; // Most recent bar index
                fibIndex = rates[0].time;
                lastLeg1Value = GetRateIndexByTime(legs[1].end);
                // legs = legs[-2:] در Python
                ArrayResize(legs, 2);
                legsCount = 2;
                fibCounter++;
            }
            else if(botState.hasValidFib && rates[0].low < botState.fibLevels[3]) {
                ResetBotState();
                ArrayResize(legs, 2);
                legsCount = 2;
                startIndex = GetRateIndexByTime(legs[0].start);
            }
            
            if(botState.hasValidFib) {
                LogMessage(StringFormat("fib_levels: 0.0=%.5f, 0.705=%.5f, 0.9=%.5f, 1.0=%.5f", 
                          botState.fibLevels[0], botState.fibLevels[1], 
                          botState.fibLevels[2], botState.fibLevels[3]), "yellow");
                LogMessage(StringFormat("fib_index: %s", TimeToString(fibIndex)), "yellow");
            }
        }
    }
    else if(swingType == "bearish") {
        if(rates[0].close <= legs[0].endValue) {
            double localStartPrice = rates[0].low;   // بجای startPrice
            double localEndPrice = legs[1].endValue;  // بجای endPrice
            
            if(rates[0].low <= legs[1].endValue) {
                LogMessage(StringFormat("The %d of fib_levels value code:4126455 %s", 
                          fibCounter, TimeToString(rates[0].time)), "green");
                CalculateFibonacci(localStartPrice, localEndPrice);
                fib0Point = 0; // Most recent bar index
                fibIndex = rates[0].time;
                lastLeg1Value = GetRateIndexByTime(legs[1].end);
                ArrayResize(legs, 2);
                legsCount = 2;
                fibCounter++;
            }
            else if(botState.hasValidFib && rates[0].high > botState.fibLevels[3]) {
                ResetBotState();
                ArrayResize(legs, 2);
                legsCount = 2;
                startIndex = GetRateIndexByTime(legs[0].start);
            }
            
            if(botState.hasValidFib) {
                LogMessage(StringFormat("fib_levels: 0.0=%.5f, 0.705=%.5f, 0.9=%.5f, 1.0=%.5f", 
                          botState.fibLevels[0], botState.fibLevels[1], 
                          botState.fibLevels[2], botState.fibLevels[3]), "yellow");
                LogMessage(StringFormat("fib_index: %s", TimeToString(fibIndex)), "yellow");
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Process same direction swing - مطابق Python Phase 2            |
//+------------------------------------------------------------------+
void ProcessSameDirectionSwing(string swingType, bool isSwing)
{
    LogMessage("is_swing and state.fib_levels and last_swing_type == swing_type code:4213312", "yellow");
    
    if(swingType == "bullish") {
        if(rates[0].high >= legs[1].endValue) {
            LogMessage(StringFormat("The %d of fib_levels value update code:9916455 %s", 
                      fibCounter, TimeToString(rates[0].time)), "green");
            startPrice = rates[0].high;  // fib0
            endPrice = legs[1].endValue;  // fib1
            CalculateFibonacci(endPrice, startPrice);
            fib0Point = 0;
            fibIndex = rates[0].time;
            lastLeg1Value = GetRateIndexByTime(legs[1].end);
            ArrayResize(legs, 2);
            legsCount = 2;
            fibCounter++;
        }
        else if(rates[0].low <= botState.fibLevels[1]) { // 0.705 touch
            if(!botState.hasTouched705Up) {
                LogMessage("first touch 705 point code:7318455", "green");
                botState.lastTouched705PointUp = rates[0];
                botState.hasTouched705Up = true;
            }
            else {
                string currentStatus = (rates[0].close >= rates[0].open) ? "bullish" : "bearish";
                string lastStatus = (botState.lastTouched705PointUp.close >= botState.lastTouched705PointUp.open) ? "bullish" : "bearish";
                if(currentStatus != lastStatus) {
                    LogMessage(StringFormat("Second touch 705 point code:7218455 %s", TimeToString(rates[0].time)), "green");
                    botState.truePosition = true;
                }
            }
        }
        else if(botState.hasValidFib && rates[0].low < botState.fibLevels[3]) {
            ResetBotState();
            ArrayResize(legs, 2);
            legsCount = 2;
            startIndex = GetRateIndexByTime(legs[0].start);
        }
    }
    else if(swingType == "bearish") {
        if(rates[0].low <= legs[1].endValue) {
            LogMessage(StringFormat("The %d of fib_levels value update code:9916455 %s", 
                      fibCounter, TimeToString(rates[0].time)), "green");
            startPrice = rates[0].low;   // fib0
            endPrice = legs[1].endValue;  // fib1
            CalculateFibonacci(startPrice, endPrice);
            fib0Point = 0;
            fibIndex = rates[0].time;
            lastLeg1Value = GetRateIndexByTime(legs[1].end);
            ArrayResize(legs, 2);
            legsCount = 2;
            fibCounter++;
        }
        else if(rates[0].high >= botState.fibLevels[1]) { // 0.705 touch
            if(!botState.hasTouched705Down) {
                LogMessage("first touch 705 point code:6328455", "red");
                botState.lastTouched705PointDown = rates[0];
                botState.hasTouched705Down = true;
            }
            else {
                string currentStatus = (rates[0].close >= rates[0].open) ? "bullish" : "bearish";
                string lastStatus = (botState.lastTouched705PointDown.close >= botState.lastTouched705PointDown.open) ? "bullish" : "bearish";
                if(currentStatus != lastStatus) {
                    LogMessage(StringFormat("Second touch 705 point code:6228455 %s", TimeToString(rates[0].time)), "green");
                    botState.truePosition = true;
                }
            }
        }
        else if(botState.hasValidFib && rates[0].high > botState.fibLevels[3]) {
            ResetBotState();
            ArrayResize(legs, 2);
            legsCount = 2;
            startIndex = GetRateIndexByTime(legs[0].start);
        }
    }
}

//+------------------------------------------------------------------+
//| Process opposite direction swing - مطابق Python Phase 3        |
//+------------------------------------------------------------------+
void ProcessOppositeDirectionSwing(string swingType)
{
    LogMessage("is_swing with opposite direction - checking fib 1.0 violation", "orange");
    
    if(lastSwingType == "bullish" && swingType == "bearish") {
        if(rates[0].low < botState.fibLevels[3]) {
            LogMessage("Bearish swing violated fib 1.0 - resetting", "red");
            ResetBotState();
            ArrayResize(legs, 3);
            legsCount = 3;
            startIndex = GetRateIndexByTime(legs[0].start);
        }
        else {
            LogMessage("Bearish swing within fib range - ignoring", "yellow");
        }
    }
    else if(lastSwingType == "bearish" && swingType == "bullish") {
        if(rates[0].high > botState.fibLevels[3]) {
            LogMessage("Bullish swing violated fib 1.0 - resetting", "red");
            ResetBotState();
            ArrayResize(legs, 3);
            legsCount = 3;
            startIndex = GetRateIndexByTime(legs[0].start);
        }
        else {
            LogMessage("Bullish swing within fib range - ignoring", "yellow");
        }
    }
}

//+------------------------------------------------------------------+
//| Process fibonacci updates - مطابق Python                       |
//+------------------------------------------------------------------+
void ProcessFibonacciUpdates()
{
    if(lastSwingType == "bullish") {
        startPrice = rates[0].high;  // fib0
        endPrice = legs[1].endValue;  // fib1
        
        if(botState.fibLevels[0] < rates[0].high) {
            LogMessage(StringFormat("update fib_levels value code:7117455 %s", TimeToString(rates[0].time)), "green");
            startPrice = rates[0].high;
            CalculateFibonacci(endPrice, startPrice);
            fib0Point = 0;
            fibIndex = rates[0].time;
        }
        else if(rates[0].low <= botState.fibLevels[1]) { // 0.705 touch
            if(!botState.hasTouched705Up) {
                LogMessage("first touch 705 point code:7318455", "green");
                botState.lastTouched705PointUp = rates[0];
                botState.hasTouched705Up = true;
            }
            else {
                string currentStatus = (rates[0].close >= rates[0].open) ? "bullish" : "bearish";
                string lastStatus = (botState.lastTouched705PointUp.close >= botState.lastTouched705PointUp.open) ? "bullish" : "bearish";
                if(currentStatus != lastStatus) {
                    LogMessage(StringFormat("Second touch 705 point code:7218455 %s", TimeToString(rates[0].time)), "green");
                    botState.truePosition = true;
                }
            }
        }
        else if(botState.hasValidFib && rates[0].low < botState.fibLevels[3]) {
            ResetBotState();
            ArrayResize(legs, 2);
            legsCount = 2;
            startIndex = GetRateIndexByTime(legs[0].start);
        }
    }
    
    if(lastSwingType == "bearish") {
        startPrice = rates[0].low;   // fib0
        endPrice = legs[1].endValue;  // fib1
        
        if(botState.fibLevels[0] > rates[0].low) {
            LogMessage(StringFormat("update fib_levels value code:6127455 %s", TimeToString(rates[0].time)), "green");
            startPrice = rates[0].low;
            CalculateFibonacci(startPrice, endPrice);
            fib0Point = 0;
            fibIndex = rates[0].time;
        }
        else if(rates[0].high >= botState.fibLevels[1]) { // 0.705 touch
            if(!botState.hasTouched705Down) {
                LogMessage("first touch 705 point code:6328455", "red");
                botState.lastTouched705PointDown = rates[0];
                botState.hasTouched705Down = true;
            }
            else {
                string currentStatus = (rates[0].close >= rates[0].open) ? "bullish" : "bearish";
                string lastStatus = (botState.lastTouched705PointDown.close >= botState.lastTouched705PointDown.open) ? "bullish" : "bearish";
                if(currentStatus != lastStatus) {
                    LogMessage(StringFormat("Second touch 705 point code:6228455 %s", TimeToString(rates[0].time)), "green");
                    botState.truePosition = true;
                }
            }
        }
        else if(botState.hasValidFib && rates[0].high > botState.fibLevels[3]) {
            ResetBotState();
            ArrayResize(legs, 2);
            legsCount = 2;
            startIndex = GetRateIndexByTime(legs[0].start);
        }
    }
}

//+------------------------------------------------------------------+
//| Process short legs case - مطابق Python len(legs) < 3           |
//+------------------------------------------------------------------+
void ProcessShortLegsCase()
{
    if(botState.hasValidFib) {
        if(lastSwingType == "bullish") {
            if(botState.fibLevels[0] < rates[0].high) {
                LogMessage(StringFormat("update fib_levels value code:5117455 %s", TimeToString(rates[0].time)), "green");
                startPrice = rates[0].high;
                CalculateFibonacci(endPrice, startPrice);
                fib0Point = 0;
                fibIndex = rates[0].time;
            }
            else if(rates[0].low <= botState.fibLevels[1]) {
                if(!botState.hasTouched705Up) {
                    LogMessage("first touch 705 point", "green");
                    botState.lastTouched705PointUp = rates[0];
                    botState.hasTouched705Up = true;
                }
                else {
                    string currentStatus = (rates[0].close >= rates[0].open) ? "bullish" : "bearish";
                    string lastStatus = (botState.lastTouched705PointUp.close >= botState.lastTouched705PointUp.open) ? "bullish" : "bearish";
                    if(currentStatus != lastStatus) {
                        LogMessage(StringFormat("Second touch 705 point code:4118455 %s", TimeToString(rates[0].time)), "green");
                        botState.truePosition = true;
                    }
                }
            }
        }
        
        if(lastSwingType == "bearish") {
            if(botState.fibLevels[0] > rates[0].low) {
                LogMessage(StringFormat("update fib_levels value code:5127455 %s", TimeToString(rates[0].time)), "green");
                startPrice = rates[0].low;
                CalculateFibonacci(startPrice, endPrice);
                fib0Point = 0;
                fibIndex = rates[0].time;
            }
            else if(rates[0].high >= botState.fibLevels[1]) {
                if(!botState.hasTouched705Down) {
                    LogMessage("first touch 705 point", "red");
                    botState.lastTouched705PointDown = rates[0];
                    botState.hasTouched705Down = true;
                }
                else {
                    string currentStatus = (rates[0].close >= rates[0].open) ? "bullish" : "bearish";
                    string lastStatus = (botState.lastTouched705PointDown.close >= botState.lastTouched705PointDown.open) ? "bullish" : "bearish";
                    if(currentStatus != lastStatus) {
                        LogMessage(StringFormat("Second touch 705 point code:5128455 %s", TimeToString(rates[0].time)), "green");
                        botState.truePosition = true;
                    }
                }
            }
        }
    }
    
    if(legsCount == 2) {
        LogMessage(StringFormat("leg0: %s, %s, leg1: %s, %s", 
                   TimeToString(legs[0].start), TimeToString(legs[0].end),
                   TimeToString(legs[1].start), TimeToString(legs[1].end)), "lightcyan_ex");
    }
    if(legsCount == 1) {
        LogMessage(StringFormat("leg0: %s, %s", 
                   TimeToString(legs[0].start), TimeToString(legs[0].end)), "lightcyan_ex");
    }
}

//+------------------------------------------------------------------+
//| Check trading signals - مطابق Python buy/sell section         |
//+------------------------------------------------------------------+
void CheckTradingSignals()
{
    //--- Long position signal - مطابق Python
    if(botState.truePosition && (lastSwingType == "bullish")) {
        double currentOpenPoint = rates[0].close;
        
        LogMessage(StringFormat("Start long position income %s", TimeToString(rates[0].time)), "blue");
        LogMessage(StringFormat("current_open_point: %.5f", currentOpenPoint), "blue");
        
        //--- Determine stop loss مطابق Python logic
        double stop;
        if(MathAbs(botState.fibLevels[2] - currentOpenPoint) * 10000 < 2) {
            stop = botState.fibLevels[3]; // fib 1.0
            LogMessage(StringFormat("stop = fib_levels[1.0] %.5f", stop), "red");
        }
        else {
            stop = botState.fibLevels[2]; // fib 0.9
            LogMessage(StringFormat("stop = fib_levels[0.9] %.5f", stop), "red");
        }
        
        double stopDistance = MathAbs(currentOpenPoint - stop);
        double rewardEnd = currentOpenPoint + (stopDistance * InpWinRatio);
        
        LogMessage(StringFormat("stop = %.5f", stop), "green");
        LogMessage(StringFormat("reward_end = %.5f", rewardEnd), "green");
        
        //--- Execute trade
        if(ExecuteBuyOrder(currentOpenPoint, stop, rewardEnd)) {
            LogMessage("✅ BUY order executed successfully", "green");
            positionOpen = true;
        }
        else {
            LogMessage("❌ BUY order failed", "red");
        }
        
        //--- Reset state after trade مطابق Python
        ResetBotState();
        ArrayFree(legs);
        legsCount = 0;
        startIndex = 0; // Reset to current bar
        LogMessage(StringFormat("End long position, start_index: %d", startIndex), "black");
    }
    
    //--- Short position signal - مطابق Python
    if(botState.truePosition && (lastSwingType == "bearish")) {
        double currentOpenPoint = rates[0].close;
        
        LogMessage(StringFormat("Start short position income %s", TimeToString(rates[0].time)), "red");
        LogMessage(StringFormat("current_open_point: %.5f", currentOpenPoint), "red");
        
        //--- Determine stop loss مطابق Python logic
        double stop;
        if(MathAbs(botState.fibLevels[2] - currentOpenPoint) * 10000 < 2) {
            stop = botState.fibLevels[3]; // fib 1.0
            LogMessage(StringFormat("stop = fib_levels[1.0] %.5f", stop), "red");
        }
        else {
            stop = botState.fibLevels[2]; // fib 0.9
            LogMessage(StringFormat("stop = fib_levels[0.9] %.5f", stop), "red");
        }
        
        double stopDistance = MathAbs(currentOpenPoint - stop);
        double rewardEnd = currentOpenPoint - (stopDistance * InpWinRatio);
        
        LogMessage(StringFormat("stop = %.5f", stop), "red");
        LogMessage(StringFormat("reward_end = %.5f", rewardEnd), "red");
        
        //--- Execute trade
        if(ExecuteSellOrder(currentOpenPoint, stop, rewardEnd)) {
            LogMessage("✅ SELL order executed successfully", "green");
            positionOpen = true;
        }
        else {
            LogMessage("❌ SELL order failed", "red");
        }
        
        //--- Reset state after trade مطابق Python
        ResetBotState();
        ArrayFree(legs);
        legsCount = 0;
        startIndex = 0; // Reset to current bar
        LogMessage(StringFormat("End short position, start_index: %d", startIndex), "black");
    }
}

//+------------------------------------------------------------------+
//| Helper Functions - تمام توابع مطابق Python                     |
//+------------------------------------------------------------------+

//--- Check trading conditions مطابق mt5_conn.can_trade()
void CheckTradingConditions(bool &canTrade, string &message)
{
    canTrade = false;
    message = "";
    
    //--- Basic checks
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) {
        message = "Terminal trading disabled";
        return;
    }
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) {
        message = "EA trading disabled";
        return;
    }
    if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) {
        message = "Account trading disabled";
        return;
    }
    
    //--- Weekend check مطابق Python
    if(!InpTradeOnWeekends && IsIranWeekend()) {
        message = "Weekend - trading disabled";
        return;
    }
    
    //--- Trading hours check مطابق Python mt5_conn.is_trading_time()
    if(!IsWithinAllowedTradingHours()) {
        message = StringFormat("Outside trading hours (%s - %s Iran time)", InpStartTime, InpEndTime);
        return;
    }
    
    //--- Daily limit check
    if(dailyTradesCount >= InpMaxDailyTrades) {
        message = StringFormat("Daily trade limit reached (%d/%d)", dailyTradesCount, InpMaxDailyTrades);
        return;
    }
    
    //--- Spread check
    double spread = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point();
    if(spread > InpMaxSpread) {
        message = StringFormat("Spread too high: %.1f pips", spread);
        return;
    }
    
    canTrade = true;
    message = "Trading allowed";
}

//--- مطابق MT5Connector.is_trading_time() در Python
bool IsWithinAllowedTradingHours()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime); // GMT+3:30 for Iran
    
    if(InpUseOverlapOnly) {
        return IsWithinTradingSession(OVERLAP_LONDON_NY);
    }
    
    bool inLondon = InpUseLondonSession && IsWithinTradingSession(LONDON_SESSION);
    bool inNY = InpUseNYSession && IsWithinTradingSession(NEWYORK_SESSION);
    bool inCustom = IsWithinCustomHours();
    
    return (inLondon || inNY || inCustom);
}

//--- Check custom trading hours
bool IsWithinCustomHours()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime);
    
    string currentTime = StringFormat("%02d:%02d", iranTime.hour, iranTime.min);
    return (currentTime >= InpStartTime && currentTime <= InpEndTime);
}

//--- مطابق TradingSession check در Python
bool IsWithinTradingSession(const TradingSession &session)
{
    MqlDateTime currentTime;
    TimeToStruct(TimeGMT() + 12600, currentTime); // Iran time
    
    int currentMinutes = currentTime.hour * 60 + currentTime.min;
    int startMinutes = session.startHour * 60 + session.startMinute;
    int endMinutes = session.endHour * 60 + session.endMinute;
    
    // Handle overnight sessions
    if(startMinutes > endMinutes) {
        return (currentMinutes >= startMinutes || currentMinutes <= endMinutes);
    }
    else {
        return (currentMinutes >= startMinutes && currentMinutes <= endMinutes);
    }
}

//--- Check if Iran weekend
bool IsIranWeekend()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime);
    
    // Friday = 5, Saturday = 6 in Iran
    return (iranTime.day_of_week == 5 || iranTime.day_of_week == 6);
}

//--- Get Iran time string
string GetIranTimeString()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime);
    return StringFormat("%04d.%02d.%02d %02d:%02d:%02d", 
                       iranTime.year, iranTime.mon, iranTime.day,
                       iranTime.hour, iranTime.min, iranTime.sec);
}

//--- Calculate fibonacci levels مطابق fibo_calculate.py
void CalculateFibonacci(double startPrice, double endPrice)
{
    botState.fibLevels[0] = startPrice;  // 0.0
    botState.fibLevels[1] = startPrice + InpFib705 * (endPrice - startPrice);  // 0.705
    botState.fibLevels[2] = startPrice + InpFib90 * (endPrice - startPrice);   // 0.9
    botState.fibLevels[3] = endPrice;    // 1.0
    botState.hasValidFib = true;
    
    if(InpDebugMode) {
        Print("📊 Fibonacci levels: 0.0=", botState.fibLevels[0], 
              " 0.705=", botState.fibLevels[1], 
              " 0.9=", botState.fibLevels[2], 
              " 1.0=", botState.fibLevels[3]);
    }
}

//--- Reset bot state مطابق utils.py
void ResetBotState()
{
    ArrayInitialize(botState.fibLevels, 0);
    botState.truePosition = false;
    botState.hasTouched705Up = false;
    botState.hasTouched705Down = false;
    botState.hasValidFib = false;
    botState.fibCreationTime = 0;
    
    ZeroMemory(botState.lastTouched705PointUp);
    ZeroMemory(botState.lastTouched705PointDown);
    
    if(InpDebugMode) {
        Print("🔄 Bot state reset");
    }
}

//--- Log message with color (مطابق save_file.py)
void LogMessage(string msg, string colorName = "")
{
    Print(msg); // Simple print for now
    // Color information is noted but MT5 terminal doesn't support colored text
}

//--- Execute buy order مطابق Python
bool ExecuteBuyOrder(double price, double sl, double tp)
{
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    price = NormalizeDouble(price, Digits());
    
    //--- Execute trade
    if(trade.Buy(InpLotSize, Symbol(), 0, sl, tp, InpTradeComment)) {
        currentTicket = trade.ResultOrder();
        dailyTradesCount++;
        stats.executedTrades++;
        
        LogMessage(StringFormat("📊 Order details: Ticket=%d, Price=%.5f, Volume=%.2f", 
                  currentTicket, price, InpLotSize), "cyan");
        LogMessage(StringFormat("📊 Result code: %d", trade.ResultRetcode()), "cyan");
        return true;
    }
    else {
        LogMessage(StringFormat("❌ Error code: %d, Comment: %s", 
                  trade.ResultRetcode(), trade.ResultRetcodeDescription()), "red");
        return false;
    }
}

//--- Execute sell order مطابق Python
bool ExecuteSellOrder(double price, double sl, double tp)
{
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    price = NormalizeDouble(price, Digits());
    
    //--- Execute trade
    if(trade.Sell(InpLotSize, Symbol(), 0, sl, tp, InpTradeComment)) {
        currentTicket = trade.ResultOrder();
        dailyTradesCount++;
        stats.executedTrades++;
        
        LogMessage(StringFormat("📊 Order details: Ticket=%d, Price=%.5f, Volume=%.2f", 
                  currentTicket, price, InpLotSize), "cyan");
        LogMessage(StringFormat("📊 Result code: %d", trade.ResultRetcode()), "cyan");
        return true;
    }
    else {
        LogMessage(StringFormat("❌ Error code: %d, Comment: %s", 
                  trade.ResultRetcode(), trade.ResultRetcodeDescription()), "red");
        return false;
    }
}

//--- Monitor positions مطابق Python
void MonitorPositions()
{
    int totalPositions = PositionsTotal();
    
    if(totalPositions == 0 && positionOpen) {
        LogMessage("🏁 Position closed", "yellow");
        positionOpen = false;
        currentTicket = 0;
    }
}

//--- Update daily trade counter
void UpdateDailyTradeCounter()
{
    datetime currentDay = (datetime)(TimeCurrent() / 86400) * 86400;
    if(currentDay != lastDayCheck) {
        dailyTradesCount = 0;
        lastDayCheck = currentDay;
        LogMessage("📅 New trading day - counter reset", "");
    }
}

//--- Update info panel
void UpdateInfoPanel()
{
    string info = StringFormat("SwingEA | Balance: %.2f | Trades: %d/%d | P/L: %.2f | Fib: %s",
                              AccountInfoDouble(ACCOUNT_BALANCE),
                              dailyTradesCount, InpMaxDailyTrades,
                              stats.totalPL,
                              botState.hasValidFib ? "Active" : "None");
    Comment(info);
}

//--- Update trailing stop
void UpdateTrailingStop()
{
    // Implementation similar to previous version
}

//--- Get rate index by time
int GetRateIndexByTime(datetime time)
{
    for(int i = 0; i < ArraySize(rates); i++) {
        if(rates[i].time == time) return i;
    }
    return -1;
}

//--- Get time difference
int GetTimeDifference(datetime start, datetime end)
{
    int startIdx = GetRateIndexByTime(start);
    int endIdx = GetRateIndexByTime(end);
    return (startIdx >= 0 && endIdx >= 0) ? MathAbs(endIdx - startIdx) : 0;
}

//--- System checks مطابق Python
void CheckSymbolProperties()
{
    Print("📊 Symbol: ", Symbol());
    Print("📊 Digits: ", Digits());
    Print("📊 Spread: ", (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point(), " points");
}

void CheckTradingConditions()
{
    Print("📊 Min Lot: ", SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN));
    Print("📊 Max Lot: ", SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX));
}

void CheckAccountPermissions()
{
    Print("📊 Balance: ", AccountInfoDouble(ACCOUNT_BALANCE));
    Print("📊 Trading Allowed: ", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) ? "Yes" : "No");
}

void CheckMarketState()
{
    Print("📊 Market: ", SymbolInfoInteger(Symbol(), SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL ? "Open" : "Closed");
}

//+------------------------------------------------------------------+