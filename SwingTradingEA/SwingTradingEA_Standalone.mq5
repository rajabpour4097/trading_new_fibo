//+------------------------------------------------------------------+
//|                                                SwingTradingEA.mq5 |
//|                              Standalone Swing Trading EA        |
//|                           No external files required            |
//+------------------------------------------------------------------+
#property copyright "Swing Trading EA"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== Trading Parameters ==="
input double   InpLotSize = 0.01;           // Lot size
input double   InpWinRatio = 1.2;           // Win ratio (TP/SL)
input int      InpThreshold = 6;            // Leg threshold (pips)
input int      InpWindowSize = 100;         // Data window size
input double   InpFib705 = 0.705;          // Fibonacci 70.5% level
input double   InpFib90 = 0.9;             // Fibonacci 90% level

input group "=== Risk Management ==="
input double   InpMaxRiskPercent = 2.0;    // Max risk per trade (% of balance)
input double   InpMaxSpread = 3.0;         // Maximum spread (pips)
input bool     InpUseTrailingStop = false; // Use trailing stop
input int      InpTrailingStopPips = 10;   // Trailing stop distance (pips)

input group "=== Time Settings ==="
input string   InpStartTime = "09:00";     // Trading start time (Iran)
input string   InpEndTime = "21:00";       // Trading end time (Iran)
input bool     InpTradeOnWeekends = false; // Trade on weekends

input group "=== Expert Settings ==="
input int      InpMagicNumber = 234000;    // Magic number
input string   InpTradeComment = "SwingEA"; // Trade comment
input bool     InpShowInfo = true;         // Show info panel
input bool     InpEnableAlerts = true;     // Enable alerts

//--- Global variables
CTrade trade;
MqlRates rates[];
datetime lastProcessedTime = 0;
int barsTotal = 0;
bool positionOpen = false;
ulong currentTicket = 0;
int dailyTradesCount = 0;
datetime lastDayCheck = 0;

//--- State variables
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

//--- Leg structure
struct Leg {
    datetime startTime;
    datetime endTime;
    double startValue;
    double endValue;
    double length;
    string direction;
};

Leg legs[];
int legsCount = 0;
string lastSwingType = "";

//--- Performance tracking
struct PerformanceStats {
    int totalSignals;
    int executedTrades;
    int profitableTrades;
    double totalPL;
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
    
    //--- Set timer
    EventSetTimer(1);
    
    //--- Array setup
    ArraySetAsSeries(rates, true);
    ArrayResize(legs, 100);
    
    //--- Print initialization
    Print("🚀 Swing Trading EA Started...");
    Print("📊 Config: Symbol=", Symbol(), ", Lot=", InpLotSize, ", Win Ratio=", InpWinRatio);
    Print("⏰ Trading Hours (Iran): ", InpStartTime, " - ", InpEndTime);
    
    //--- System checks
    CheckSystemRequirements();
    
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
    //--- Check new bar
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
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(!CanTrade()) return;
    
    if(positionOpen) {
        ManageOpenPosition();
    }
}

//+------------------------------------------------------------------+
//| Main data processing                                             |
//+------------------------------------------------------------------+
void ProcessNewData()
{
    //--- Get historical data
    int copied = CopyRates(Symbol(), PERIOD_M1, 0, InpWindowSize, rates);
    if(copied <= 0) {
        Print("❌ Failed to get historical data");
        return;
    }
    
    //--- Check for new data
    datetime currentTime = rates[0].time;
    if(lastProcessedTime != 0 && currentTime == lastProcessedTime) {
        return;
    }
    
    lastProcessedTime = currentTime;
    
    //--- Detect legs
    DetectLegs();
    
    //--- Process swings
    if(legsCount >= 3) {
        ProcessSwings();
    }
    
    //--- Update fibonacci
    if(botState.hasValidFib) {
        UpdateFibonacci();
    }
    
    //--- Check signals
    CheckTradingSignals();
    
    //--- Monitor positions
    MonitorPositions();
}

//+------------------------------------------------------------------+
//| Simplified leg detection                                         |
//+------------------------------------------------------------------+
void DetectLegs()
{
    legsCount = 0;
    ArrayFree(legs);
    
    double threshold = InpThreshold;
    int minBars = 3;
    
    //--- Simple leg detection based on significant price moves
    for(int i = 20; i < ArraySize(rates) - 20; i += 10) {
        double high1 = rates[i].high;
        double low1 = rates[i].low;
        double high2 = rates[i + 10].high;
        double low2 = rates[i + 10].low;
        
        double priceDiff = MathMax(
            MathAbs(high1 - low2) * 10000,
            MathAbs(high2 - low1) * 10000
        );
        
        if(priceDiff >= threshold && legsCount < 50) {
            ArrayResize(legs, legsCount + 1);
            legs[legsCount].startTime = rates[i + 10].time;
            legs[legsCount].endTime = rates[i].time;
            legs[legsCount].startValue = (high1 > low2) ? low2 : high2;
            legs[legsCount].endValue = (high1 > low2) ? high1 : low1;
            legs[legsCount].length = priceDiff;
            legs[legsCount].direction = (high1 > low2) ? "up" : "down";
            legsCount++;
        }
    }
    
    if(legsCount > 0) {
        Print("📈 Detected ", legsCount, " legs");
    }
}

//+------------------------------------------------------------------+
//| Process swing patterns                                           |
//+------------------------------------------------------------------+
void ProcessSwings()
{
    if(legsCount < 3) return;
    
    //--- Take last 3 legs
    Leg lastLegs[3];
    for(int i = 0; i < 3; i++) {
        lastLegs[i] = legs[legsCount - 3 + i];
    }
    
    string swingType = "";
    bool isSwing = false;
    
    //--- Check bullish swing
    if(lastLegs[1].endValue > lastLegs[0].startValue && 
       lastLegs[0].endValue > lastLegs[1].endValue) {
        swingType = "bullish";
        isSwing = true;
    }
    //--- Check bearish swing
    else if(lastLegs[1].endValue < lastLegs[0].startValue && 
            lastLegs[0].endValue < lastLegs[1].endValue) {
        swingType = "bearish";
        isSwing = true;
    }
    
    if(isSwing) {
        stats.totalSignals++;
        lastSwingType = swingType;
        
        Print("✅ Swing detected: ", swingType);
        
        //--- Create fibonacci if not exists
        if(!botState.hasValidFib) {
            CreateFibonacci(swingType);
        }
        
        //--- Send alert
        if(InpEnableAlerts) {
            Alert("Swing detected: ", swingType, " on ", Symbol());
        }
    }
}

//+------------------------------------------------------------------+
//| Create fibonacci levels                                          |
//+------------------------------------------------------------------+
void CreateFibonacci(string swingType)
{
    double startPrice, endPrice;
    
    if(swingType == "bullish") {
        // For bullish swing: fib from low to high
        endPrice = rates[0].high;
        startPrice = rates[10].low; // Look back 10 bars
        
        // Find recent low
        for(int i = 0; i < 20; i++) {
            if(rates[i].low < startPrice) {
                startPrice = rates[i].low;
            }
        }
    }
    else {
        // For bearish swing: fib from high to low
        startPrice = rates[0].low;
        endPrice = rates[10].high; // Look back 10 bars
        
        // Find recent high
        for(int i = 0; i < 20; i++) {
            if(rates[i].high > endPrice) {
                endPrice = rates[i].high;
            }
        }
    }
    
    //--- Calculate fibonacci levels
    botState.fibLevels[0] = startPrice;  // 0.0
    botState.fibLevels[1] = startPrice + InpFib705 * (endPrice - startPrice);  // 0.705
    botState.fibLevels[2] = startPrice + InpFib90 * (endPrice - startPrice);   // 0.9
    botState.fibLevels[3] = endPrice;    // 1.0
    botState.hasValidFib = true;
    botState.fibCreationTime = TimeCurrent();
    
    Print("📊 Fibonacci created for ", swingType, " swing:");
    Print("   0.0% = ", botState.fibLevels[0]);
    Print("   70.5% = ", botState.fibLevels[1]);
    Print("   90.0% = ", botState.fibLevels[2]);
    Print("   100.0% = ", botState.fibLevels[3]);
}

//+------------------------------------------------------------------+
//| Update fibonacci levels                                          |
//+------------------------------------------------------------------+
void UpdateFibonacci()
{
    if(!botState.hasValidFib) return;
    
    double currentHigh = rates[0].high;
    double currentLow = rates[0].low;
    
    //--- Check for fibonacci touch
    if(currentLow <= botState.fibLevels[1] && currentHigh >= botState.fibLevels[1]) {
        HandleFibonacciTouch();
    }
    
    //--- Check for fibonacci violation
    if((lastSwingType == "bullish" && currentLow < botState.fibLevels[3]) ||
       (lastSwingType == "bearish" && currentHigh > botState.fibLevels[3])) {
        Print("❌ Fibonacci levels violated - resetting");
        ResetBotState();
    }
}

//+------------------------------------------------------------------+
//| Handle fibonacci touch                                           |
//+------------------------------------------------------------------+
void HandleFibonacciTouch()
{
    bool currentIsBullish = (rates[0].close >= rates[0].open);
    
    if(lastSwingType == "bullish") {
        if(!botState.hasTouched705Up) {
            Print("🎯 First touch of 0.705 level (bullish)");
            botState.lastTouched705PointUp = rates[0];
            botState.hasTouched705Up = true;
        }
        else {
            bool lastWasBullish = (botState.lastTouched705PointUp.close >= botState.lastTouched705PointUp.open);
            if(currentIsBullish != lastWasBullish) {
                Print("✅ Signal: Second touch with different candle type!");
                botState.truePosition = true;
            }
        }
    }
    else if(lastSwingType == "bearish") {
        if(!botState.hasTouched705Down) {
            Print("🎯 First touch of 0.705 level (bearish)");
            botState.lastTouched705PointDown = rates[0];
            botState.hasTouched705Down = true;
        }
        else {
            bool lastWasBullish = (botState.lastTouched705PointDown.close >= botState.lastTouched705PointDown.open);
            if(currentIsBullish != lastWasBullish) {
                Print("✅ Signal: Second touch with different candle type!");
                botState.truePosition = true;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check trading signals                                            |
//+------------------------------------------------------------------+
void CheckTradingSignals()
{
    if(!botState.truePosition || !botState.hasValidFib) return;
    if(positionOpen || !CanTrade()) return;
    
    //--- Check spread
    double spread = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point();
    if(spread > InpMaxSpread) return;
    
    //--- Execute trade
    if(lastSwingType == "bullish") {
        ExecuteBuyOrder();
    }
    else if(lastSwingType == "bearish") {
        ExecuteSellOrder();
    }
}

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder()
{
    double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    double sl, tp;
    
    //--- Calculate stop loss
    if(MathAbs(botState.fibLevels[2] - entryPrice) * 10000 < 2) {
        sl = botState.fibLevels[3]; // Use fib 1.0
    }
    else {
        sl = botState.fibLevels[2]; // Use fib 0.9
    }
    
    //--- Calculate take profit
    double stopDistance = MathAbs(entryPrice - sl);
    tp = entryPrice + (stopDistance * InpWinRatio);
    
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    
    //--- Execute trade
    if(trade.Buy(InpLotSize, Symbol(), entryPrice, sl, tp, InpTradeComment)) {
        currentTicket = trade.ResultOrder();
        positionOpen = true;
        dailyTradesCount++;
        stats.executedTrades++;
        
        Print("✅ BUY executed: Price=", entryPrice, " SL=", sl, " TP=", tp);
        
        ResetBotState();
    }
    else {
        Print("❌ BUY failed: ", trade.ResultRetcodeDescription());
        botState.consecutiveFailures++;
    }
}

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder()
{
    double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double sl, tp;
    
    //--- Calculate stop loss
    if(MathAbs(botState.fibLevels[2] - entryPrice) * 10000 < 2) {
        sl = botState.fibLevels[3]; // Use fib 1.0
    }
    else {
        sl = botState.fibLevels[2]; // Use fib 0.9
    }
    
    //--- Calculate take profit
    double stopDistance = MathAbs(entryPrice - sl);
    tp = entryPrice - (stopDistance * InpWinRatio);
    
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    
    //--- Execute trade
    if(trade.Sell(InpLotSize, Symbol(), entryPrice, sl, tp, InpTradeComment)) {
        currentTicket = trade.ResultOrder();
        positionOpen = true;
        dailyTradesCount++;
        stats.executedTrades++;
        
        Print("✅ SELL executed: Price=", entryPrice, " SL=", sl, " TP=", tp);
        
        ResetBotState();
    }
    else {
        Print("❌ SELL failed: ", trade.ResultRetcodeDescription());
        botState.consecutiveFailures++;
    }
}

//+------------------------------------------------------------------+
//| Helper Functions                                                 |
//+------------------------------------------------------------------+

bool CanTrade()
{
    //--- Basic checks
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ||
       !MQLInfoInteger(MQL_TRADE_ALLOWED) ||
       !AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) {
        return false;
    }
    
    //--- Weekend check
    if(!InpTradeOnWeekends) {
        MqlDateTime dt;
        TimeToStruct(TimeCurrent(), dt);
        if(dt.day_of_week == 0 || dt.day_of_week == 6) return false;
    }
    
    //--- Trading hours check (Iran time)
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime); // GMT+3:30
    string currentTime = StringFormat("%02d:%02d", iranTime.hour, iranTime.min);
    
    if(currentTime < InpStartTime || currentTime > InpEndTime) {
        return false;
    }
    
    //--- Daily limit check
    if(dailyTradesCount >= 10) return false;
    
    return true;
}

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
    
    Print("🔄 Bot state reset");
}

void UpdateDailyTradeCounter()
{
    datetime currentDay = (datetime)(TimeCurrent() / 86400) * 86400;
    if(currentDay != lastDayCheck) {
        dailyTradesCount = 0;
        lastDayCheck = currentDay;
        Print("📅 New trading day - counter reset");
    }
}

void UpdateInfoPanel()
{
    string info = StringFormat("SwingEA | Balance: %.2f | Trades: %d | P/L: %.2f",
                              AccountInfoDouble(ACCOUNT_BALANCE),
                              dailyTradesCount,
                              stats.totalPL);
    Comment(info);
}

void ManageOpenPosition()
{
    if(!PositionSelectByTicket(currentTicket)) {
        positionOpen = false;
        currentTicket = 0;
        return;
    }
    
    double profit = PositionGetDouble(POSITION_PROFIT);
    stats.totalPL += profit;
    
    if(profit > 0) {
        stats.profitableTrades++;
    }
}

void UpdateTrailingStop()
{
    if(!PositionSelectByTicket(currentTicket)) return;
    
    double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
    double currentPrice = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 
                         SymbolInfoDouble(Symbol(), SYMBOL_BID) : 
                         SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    
    double currentSL = PositionGetDouble(POSITION_SL);
    double newSL = currentSL;
    
    if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
        newSL = currentPrice - InpTrailingStopPips * Point() * 10;
        if(newSL > currentSL + Point() * 10) {
            trade.PositionModify(currentTicket, newSL, PositionGetDouble(POSITION_TP));
        }
    }
    else {
        newSL = currentPrice + InpTrailingStopPips * Point() * 10;
        if(newSL < currentSL - Point() * 10) {
            trade.PositionModify(currentTicket, newSL, PositionGetDouble(POSITION_TP));
        }
    }
}

void MonitorPositions()
{
    if(PositionsTotal() == 0 && positionOpen) {
        Print("🏁 Position closed");
        positionOpen = false;
        currentTicket = 0;
    }
}

void CheckSystemRequirements()
{
    Print("📊 System Check:");
    Print("   Symbol: ", Symbol());
    Print("   Digits: ", Digits());
    Print("   Spread: ", (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point(), " points");
    Print("   Min Lot: ", SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN));
    Print("   Max Lot: ", SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX));
    Print("   Trading Allowed: ", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) ? "Yes" : "No");
}

//+------------------------------------------------------------------+
