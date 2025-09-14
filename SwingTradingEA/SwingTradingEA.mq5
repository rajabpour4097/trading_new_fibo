//+------------------------------------------------------------------+
//|                                                SwingTradingEA.mq5 |
//|                              Converted from main_metatrader.py   |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input double   InpLotSize = 0.01;           // Lot size
input double   InpWinRatio = 1.2;           // Win ratio (TP/SL)
input int      InpThreshold = 6;            // Leg threshold (pips)
input int      InpWindowSize = 100;         // Data window size
input int      InpMinSwingSize = 4;         // Minimum swing size
input double   InpFib705 = 0.705;          // Fibonacci 70.5% level
input double   InpFib90 = 0.9;             // Fibonacci 90% level
input int      InpMagicNumber = 234000;    // Magic number
input string   InpStartTime = "09:00";     // Trading start time (Iran)
input string   InpEndTime = "21:00";       // Trading end time (Iran)
input bool     InpTradeOnWeekends = false; // Trade on weekends
input double   InpMaxSpread = 3.0;         // Maximum spread (pips)
input double   InpEntryTolerance = 2.0;    // Entry tolerance (pips)

//--- Global variables
CTrade trade;
MqlRates rates[];
datetime lastProcessedTime = 0;
int barsTotal = 0;
bool positionOpen = false;

//--- State variables (equivalent to BotState in Python)
struct BotState {
    double fibLevels[4];        // [0.0, 0.705, 0.9, 1.0]
    bool truePosition;
    MqlRates lastTouched705PointUp;
    MqlRates lastTouched705PointDown;
    bool hasTouched705Up;
    bool hasTouched705Down;
    bool hasValidFib;
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
int startIndex = 0;
datetime fibIndex = 0;
int fib0Point = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Set magic number
    trade.SetExpertMagicNumber(InpMagicNumber);
    
    //--- Initialize bot state
    ResetBotState();
    
    //--- Set timer for processing
    EventSetTimer(1); // Check every second
    
    //--- Print initialization message
    Print("🚀 Swing Trading EA Started...");
    Print("📊 Config: Symbol=", Symbol(), ", Lot=", InpLotSize, ", Win Ratio=", InpWinRatio);
    Print("⏰ Trading Hours (Iran): ", InpStartTime, " - ", InpEndTime);
    
    //--- Check trading conditions
    CheckSymbolProperties();
    CheckTradingLimits();
    CheckAccountPermissions();
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    Print("🔌 Swing Trading EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check if new bar formed
    int bars = iBars(Symbol(), PERIOD_M1);
    if(bars <= barsTotal) return;
    
    //--- Process new data
    ProcessNewData();
    barsTotal = bars;
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    //--- Check trading hours
    if(!CanTrade()) return;
    
    //--- Check for new data every second (similar to Python version)
    ProcessNewData();
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool CanTrade()
{
    //--- Check weekend
    if(!InpTradeOnWeekends && (DayOfWeek() == 0 || DayOfWeek() == 6)) {
        return false;
    }
    
    //--- Check trading hours (Iran time)
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime); // GMT+3:30 for Iran
    
    string currentTime = StringFormat("%02d:%02d", iranTime.hour, iranTime.min);
    
    if(currentTime >= InpStartTime && currentTime <= InpEndTime) {
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Process new market data                                          |
//+------------------------------------------------------------------+
void ProcessNewData()
{
    //--- Get historical data
    if(CopyRates(Symbol(), PERIOD_M1, 0, InpWindowSize * 2, rates) <= 0) {
        Print("❌ Failed to get historical data");
        return;
    }
    
    int dataSize = ArraySize(rates);
    if(dataSize < InpWindowSize) return;
    
    //--- Check if new data arrived
    datetime currentTime = rates[dataSize-1].time;
    if(lastProcessedTime != 0 && currentTime == lastProcessedTime) {
        return; // No new data
    }
    
    Print("📊 Processing new data: ", TimeToString(currentTime));
    lastProcessedTime = currentTime;
    
    //--- Get legs from current data
    GetLegs(startIndex, dataSize-1);
    
    //--- Process legs
    if(legsCount > 2) {
        // Take last 3 legs
        Leg lastThreeLegs[];
        ArrayResize(lastThreeLegs, 3);
        for(int i = 0; i < 3; i++) {
            lastThreeLegs[i] = legs[legsCount - 3 + i];
        }
        
        string swingType;
        bool isSwing = GetSwingPoints(lastThreeLegs, swingType);
        
        ProcessSwingLogic(swingType, isSwing, dataSize-1);
    }
    else if(legsCount < 3 && botState.hasValidFib) {
        ProcessFibonacciUpdates(dataSize-1);
    }
    
    //--- Check for trading signals
    CheckTradingSignals(dataSize-1);
    
    //--- Monitor open positions
    MonitorPositions();
}

//+------------------------------------------------------------------+
//| Get legs from price data                                         |
//+------------------------------------------------------------------+
void GetLegs(int startIdx, int endIdx)
{
    ArrayFree(legs);
    legsCount = 0;
    
    if(endIdx - startIdx < 3) return;
    
    datetime currentStartIndex = rates[startIdx].time;
    int j = 0;
    
    for(int i = startIdx + 1; i <= endIdx; i++) {
        //--- Determine current price based on candle direction
        bool currentIsBullish = (rates[i].close >= rates[i].open);
        double currentPrice;
        
        if(j > 0 && legs[j-1].direction == "up" && rates[i].high >= rates[i-1].high) {
            currentPrice = rates[i].high;
        }
        else if(j > 0 && legs[j-1].direction == "down" && rates[i].low <= rates[i-1].low) {
            currentPrice = rates[i].low;
        }
        else {
            currentPrice = currentIsBullish ? rates[i].high : rates[i].low;
        }
        
        //--- Determine start price
        int startIndexPos = GetRateIndex(currentStartIndex);
        if(startIndexPos < 0) continue;
        
        bool startIsBullish = (rates[startIndexPos].close >= rates[startIndexPos].open);
        double startPrice = startIsBullish ? rates[startIndexPos].high : rates[startIndexPos].low;
        
        //--- Calculate price difference in pips
        double priceDiff = MathAbs(currentPrice - startPrice) * 10000;
        
        //--- Determine direction
        string direction = "";
        if(rates[i].close >= rates[startIndexPos].close || 
           (rates[i].high > rates[i-1].high && rates[i].close >= rates[i-1].close)) {
            direction = "up";
        }
        else if(rates[i].close < rates[startIndexPos].close || 
                (rates[i].low < rates[i-1].low && rates[i].close < rates[i-1].close)) {
            direction = "down";
        }
        
        //--- Check if threshold is met
        if(priceDiff >= InpThreshold && priceDiff < InpThreshold * 5) {
            if(j > 0 && legs[j-1].direction == direction) {
                // Extend existing leg
                legs[j-1].endTime = rates[i].time;
                legs[j-1].endValue = currentPrice;
                legs[j-1].length = priceDiff + legs[j-1].length;
                currentStartIndex = rates[i].time;
            }
            else if(GetTimeDifference(currentStartIndex, rates[i].time) >= 3) {
                // Create new leg
                ArrayResize(legs, j + 1);
                legs[j].startTime = currentStartIndex;
                legs[j].startValue = startPrice;
                legs[j].endTime = rates[i].time;
                legs[j].endValue = currentPrice;
                legs[j].length = priceDiff;
                legs[j].direction = direction;
                
                j++;
                currentStartIndex = rates[i].time;
            }
        }
        else if(j > 0 && priceDiff < InpThreshold) {
            // Update existing leg if price continues in same direction
            if((legs[j-1].direction == "up" && rates[i].high >= GetHighAtTime(currentStartIndex)) ||
               (legs[j-1].direction == "down" && rates[i].low <= GetLowAtTime(currentStartIndex))) {
                
                legs[j-1].endTime = rates[i].time;
                legs[j-1].endValue = currentPrice;
                currentStartIndex = rates[i].time;
            }
        }
    }
    
    legsCount = j;
    Print("Found ", legsCount, " legs");
}

//+------------------------------------------------------------------+
//| Get swing points analysis                                        |
//+------------------------------------------------------------------+
bool GetSwingPoints(Leg &lastLegs[], string &swingType)
{
    if(ArraySize(lastLegs) != 3) return false;
    
    swingType = "";
    bool isSwing = false;
    
    //--- Check for bullish swing
    if(lastLegs[1].endValue > lastLegs[0].startValue && 
       lastLegs[0].endValue > lastLegs[1].endValue) {
        
        // Verify with candle analysis
        int trueCandles = 0;
        bool firstCandle = false;
        double lastCandleClose = 0;
        
        int startIdx = GetRateIndex(lastLegs[1].startTime);
        int endIdx = GetRateIndex(lastLegs[1].endTime);
        
        for(int k = startIdx; k <= endIdx; k++) {
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
    //--- Check for bearish swing
    else if(lastLegs[1].endValue < lastLegs[0].startValue && 
            lastLegs[0].endValue < lastLegs[1].endValue) {
        
        // Verify with candle analysis
        int trueCandles = 0;
        bool firstCandle = false;
        double lastCandleClose = 0;
        
        int startIdx = GetRateIndex(lastLegs[1].startTime);
        int endIdx = GetRateIndex(lastLegs[1].endTime);
        
        for(int k = startIdx; k <= endIdx; k++) {
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
    
    return isSwing;
}

//+------------------------------------------------------------------+
//| Calculate Fibonacci retracement levels                          |
//+------------------------------------------------------------------+
void CalculateFibonacci(double startPrice, double endPrice)
{
    botState.fibLevels[0] = startPrice;  // 0.0
    botState.fibLevels[1] = startPrice + InpFib705 * (endPrice - startPrice);  // 0.705
    botState.fibLevels[2] = startPrice + InpFib90 * (endPrice - startPrice);   // 0.9
    botState.fibLevels[3] = endPrice;    // 1.0
    botState.hasValidFib = true;
    
    Print("📊 Fibonacci levels: 0.0=", botState.fibLevels[0], 
          " 0.705=", botState.fibLevels[1], 
          " 0.9=", botState.fibLevels[2], 
          " 1.0=", botState.fibLevels[3]);
}

//+------------------------------------------------------------------+
//| Process swing logic                                              |
//+------------------------------------------------------------------+
void ProcessSwingLogic(string swingType, bool isSwing, int currentIdx)
{
    if(!isSwing && !botState.hasValidFib) return;
    
    double currentClose = rates[currentIdx].close;
    double currentHigh = rates[currentIdx].high;
    double currentLow = rates[currentIdx].low;
    
    //--- Phase 1: Initial swing detection
    if(isSwing && !botState.hasValidFib) {
        Print("🔍 New swing detected: ", swingType);
        lastSwingType = swingType;
        
        if(swingType == "bullish") {
            if(currentClose >= legs[legsCount-3].endValue) {
                double startPrice = currentHigh;  // fib 0
                double endPrice = legs[legsCount-2].endValue;  // fib 1
                
                if(currentHigh >= legs[legsCount-2].endValue) {
                    Print("✅ Creating bullish fibonacci levels");
                    CalculateFibonacci(endPrice, startPrice);
                    fibIndex = rates[currentIdx].time;
                }
            }
        }
        else if(swingType == "bearish") {
            if(currentClose <= legs[legsCount-3].endValue) {
                double startPrice = currentLow;   // fib 0
                double endPrice = legs[legsCount-2].endValue;  // fib 1
                
                if(currentLow <= legs[legsCount-2].endValue) {
                    Print("✅ Creating bearish fibonacci levels");
                    CalculateFibonacci(startPrice, endPrice);
                    fibIndex = rates[currentIdx].time;
                }
            }
        }
    }
    //--- Phase 2: Update existing fibonacci in same direction
    else if(isSwing && botState.hasValidFib && lastSwingType == swingType) {
        Print("🔄 Updating fibonacci in same direction: ", swingType);
        
        if(swingType == "bullish") {
            if(currentHigh >= legs[legsCount-2].endValue) {
                double startPrice = currentHigh;
                double endPrice = legs[legsCount-2].endValue;
                CalculateFibonacci(endPrice, startPrice);
                fibIndex = rates[currentIdx].time;
            }
            else if(currentLow <= botState.fibLevels[1]) { // 0.705 level
                HandleFibonacciTouch(currentIdx, true);
            }
            else if(currentLow < botState.fibLevels[3]) { // Below 1.0 level
                Print("❌ Price broke below fib 1.0 - resetting");
                ResetBotState();
            }
        }
        else if(swingType == "bearish") {
            if(currentLow <= legs[legsCount-2].endValue) {
                double startPrice = currentLow;
                double endPrice = legs[legsCount-2].endValue;
                CalculateFibonacci(startPrice, endPrice);
                fibIndex = rates[currentIdx].time;
            }
            else if(currentHigh >= botState.fibLevels[1]) { // 0.705 level
                HandleFibonacciTouch(currentIdx, false);
            }
            else if(currentHigh > botState.fibLevels[3]) { // Above 1.0 level
                Print("❌ Price broke above fib 1.0 - resetting");
                ResetBotState();
            }
        }
    }
    //--- Phase 3: Opposite swing handling
    else if(isSwing && botState.hasValidFib && lastSwingType != swingType) {
        Print("⚠️ Opposite swing detected - checking fib 1.0 violation");
        
        if(lastSwingType == "bullish" && swingType == "bearish") {
            if(currentLow < botState.fibLevels[3]) {
                Print("❌ Bearish swing violated fib 1.0 - resetting");
                ResetBotState();
            }
        }
        else if(lastSwingType == "bearish" && swingType == "bullish") {
            if(currentHigh > botState.fibLevels[3]) {
                Print("❌ Bullish swing violated fib 1.0 - resetting");
                ResetBotState();
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Handle fibonacci level touches                                   |
//+------------------------------------------------------------------+
void HandleFibonacciTouch(int currentIdx, bool isBullish)
{
    string currentStatus = (rates[currentIdx].close >= rates[currentIdx].open) ? "bullish" : "bearish";
    
    if(isBullish) {
        if(!botState.hasTouched705Up) {
            Print("🎯 First touch of 0.705 level (bullish)");
            botState.lastTouched705PointUp = rates[currentIdx];
            botState.hasTouched705Up = true;
        }
        else {
            string lastStatus = (botState.lastTouched705PointUp.close >= botState.lastTouched705PointUp.open) ? "bullish" : "bearish";
            if(currentStatus != lastStatus) {
                Print("✅ Second touch of 0.705 level with different candle type - SIGNAL!");
                botState.truePosition = true;
            }
        }
    }
    else {
        if(!botState.hasTouched705Down) {
            Print("🎯 First touch of 0.705 level (bearish)");
            botState.lastTouched705PointDown = rates[currentIdx];
            botState.hasTouched705Down = true;
        }
        else {
            string lastStatus = (botState.lastTouched705PointDown.close >= botState.lastTouched705PointDown.open) ? "bullish" : "bearish";
            if(currentStatus != lastStatus) {
                Print("✅ Second touch of 0.705 level with different candle type - SIGNAL!");
                botState.truePosition = true;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Process fibonacci updates without swing                         |
//+------------------------------------------------------------------+
void ProcessFibonacciUpdates(int currentIdx)
{
    if(!botState.hasValidFib) return;
    
    double currentHigh = rates[currentIdx].high;
    double currentLow = rates[currentIdx].low;
    
    if(lastSwingType == "bullish") {
        if(botState.fibLevels[0] < currentHigh) {
            Print("🔄 Updating bullish fibonacci - new high");
            double endPrice = botState.fibLevels[3]; // Keep fib 1.0
            CalculateFibonacci(endPrice, currentHigh);
            fibIndex = rates[currentIdx].time;
        }
        else if(currentLow <= botState.fibLevels[1]) {
            HandleFibonacciTouch(currentIdx, true);
        }
        else if(currentLow < botState.fibLevels[3]) {
            Print("❌ Price broke below fib 1.0 - resetting");
            ResetBotState();
        }
    }
    else if(lastSwingType == "bearish") {
        if(botState.fibLevels[0] > currentLow) {
            Print("🔄 Updating bearish fibonacci - new low");
            double endPrice = botState.fibLevels[3]; // Keep fib 1.0
            CalculateFibonacci(currentLow, endPrice);
            fibIndex = rates[currentIdx].time;
        }
        else if(currentHigh >= botState.fibLevels[1]) {
            HandleFibonacciTouch(currentIdx, false);
        }
        else if(currentHigh > botState.fibLevels[3]) {
            Print("❌ Price broke above fib 1.0 - resetting");
            ResetBotState();
        }
    }
}

//+------------------------------------------------------------------+
//| Check for trading signals and execute trades                    |
//+------------------------------------------------------------------+
void CheckTradingSignals(int currentIdx)
{
    if(!botState.truePosition || !botState.hasValidFib) return;
    
    double currentPrice = rates[currentIdx].close;
    double stopLoss, takeProfit;
    
    //--- Long position signal
    if((lastSwingType == "bullish") && !positionOpen) {
        Print("📈 Long position signal detected");
        
        // Determine stop loss
        if(MathAbs(botState.fibLevels[2] - currentPrice) * 10000 < 2) {
            stopLoss = botState.fibLevels[3]; // Use fib 1.0
        }
        else {
            stopLoss = botState.fibLevels[2]; // Use fib 0.9
        }
        
        double stopDistance = MathAbs(currentPrice - stopLoss);
        takeProfit = currentPrice + (stopDistance * InpWinRatio);
        
        Print("🎯 Long trade: Entry=", currentPrice, " SL=", stopLoss, " TP=", takeProfit);
        
        if(OpenBuyPosition(currentPrice, stopLoss, takeProfit)) {
            positionOpen = true;
            ResetBotState();
        }
    }
    //--- Short position signal
    else if((lastSwingType == "bearish") && !positionOpen) {
        Print("📉 Short position signal detected");
        
        // Determine stop loss
        if(MathAbs(botState.fibLevels[2] - currentPrice) * 10000 < 2) {
            stopLoss = botState.fibLevels[3]; // Use fib 1.0
        }
        else {
            stopLoss = botState.fibLevels[2]; // Use fib 0.9
        }
        
        double stopDistance = MathAbs(currentPrice - stopLoss);
        takeProfit = currentPrice - (stopDistance * InpWinRatio);
        
        Print("🎯 Short trade: Entry=", currentPrice, " SL=", stopLoss, " TP=", takeProfit);
        
        if(OpenSellPosition(currentPrice, stopLoss, takeProfit)) {
            positionOpen = true;
            ResetBotState();
        }
    }
}

//+------------------------------------------------------------------+
//| Open buy position                                                |
//+------------------------------------------------------------------+
bool OpenBuyPosition(double price, double sl, double tp)
{
    //--- Check spread
    double spread = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point();
    if(spread > InpMaxSpread) {
        Print("❌ Spread too high: ", spread, " pips");
        return false;
    }
    
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    
    //--- Execute trade
    if(trade.Buy(InpLotSize, Symbol(), 0, sl, tp, "Bullish Swing")) {
        Print("✅ BUY order executed: Ticket=", trade.ResultOrder());
        return true;
    }
    else {
        Print("❌ BUY order failed: ", trade.ResultRetcodeDescription());
        return false;
    }
}

//+------------------------------------------------------------------+
//| Open sell position                                               |
//+------------------------------------------------------------------+
bool OpenSellPosition(double price, double sl, double tp)
{
    //--- Check spread
    double spread = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point();
    if(spread > InpMaxSpread) {
        Print("❌ Spread too high: ", spread, " pips");
        return false;
    }
    
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    
    //--- Execute trade
    if(trade.Sell(InpLotSize, Symbol(), 0, sl, tp, "Bearish Swing")) {
        Print("✅ SELL order executed: Ticket=", trade.ResultOrder());
        return true;
    }
    else {
        Print("❌ SELL order failed: ", trade.ResultRetcodeDescription());
        return false;
    }
}

//+------------------------------------------------------------------+
//| Monitor open positions                                           |
//+------------------------------------------------------------------+
void MonitorPositions()
{
    int totalPositions = PositionsTotal();
    
    if(totalPositions == 0 && positionOpen) {
        Print("🏁 Position closed");
        positionOpen = false;
    }
}

//+------------------------------------------------------------------+
//| Reset bot state                                                  |
//+------------------------------------------------------------------+
void ResetBotState()
{
    ArrayInitialize(botState.fibLevels, 0);
    botState.truePosition = false;
    botState.hasTouched705Up = false;
    botState.hasTouched705Down = false;
    botState.hasValidFib = false;
    
    // Clear touched points
    ZeroMemory(botState.lastTouched705PointUp);
    ZeroMemory(botState.lastTouched705PointDown);
    
    Print("🔄 Bot state reset");
}

//+------------------------------------------------------------------+
//| Helper functions                                                 |
//+------------------------------------------------------------------+
int GetRateIndex(datetime time)
{
    for(int i = ArraySize(rates) - 1; i >= 0; i--) {
        if(rates[i].time == time) return i;
    }
    return -1;
}

double GetHighAtTime(datetime time)
{
    int idx = GetRateIndex(time);
    return (idx >= 0) ? rates[idx].high : 0;
}

double GetLowAtTime(datetime time)
{
    int idx = GetRateIndex(time);
    return (idx >= 0) ? rates[idx].low : 0;
}

int GetTimeDifference(datetime start, datetime end)
{
    int startIdx = GetRateIndex(start);
    int endIdx = GetRateIndex(end);
    return (startIdx >= 0 && endIdx >= 0) ? (endIdx - startIdx) : 0;
}

void CheckSymbolProperties()
{
    Print("📊 Symbol: ", Symbol());
    Print("📊 Digits: ", Digits());
    Print("📊 Point: ", Point());
    Print("📊 Spread: ", (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point(), " pips");
}

void CheckTradingLimits()
{
    double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    
    Print("📊 Min Lot: ", minLot, ", Max Lot: ", maxLot, ", Step: ", stepLot);
    
    if(InpLotSize < minLot || InpLotSize > maxLot) {
        Print("⚠️ WARNING: Lot size ", InpLotSize, " is outside broker limits");
    }
}

void CheckAccountPermissions()
{
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) {
        Print("❌ Trading is not allowed in terminal");
    }
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) {
        Print("❌ Trading is not allowed for EA");
    }
    if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) {
        Print("❌ Trading is not allowed for account");
    }
    if(!AccountInfoInteger(ACCOUNT_TRADE_EXPERT)) {
        Print("❌ Expert trading is not allowed for account");
    }
    
    Print("💰 Account Balance: ", AccountInfoDouble(ACCOUNT_BALANCE));
    Print("💰 Account Equity: ", AccountInfoDouble(ACCOUNT_EQUITY));
    Print("💰 Account Margin: ", AccountInfoDouble(ACCOUNT_MARGIN));
}
//+------------------------------------------------------------------+
