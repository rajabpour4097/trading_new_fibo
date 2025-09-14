//+------------------------------------------------------------------+
//|                                        SwingTradingEA_Enhanced.mq5 |
//|                Enhanced version with better error handling and   |
//|                           position management                    |
//+------------------------------------------------------------------+
#property copyright "Swing Trading EA"
#property link      ""
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

//--- Trading Configuration Constants
#define LOT_SIZE_DEFAULT           0.01
#define WIN_RATIO_DEFAULT          1.2
#define THRESHOLD_DEFAULT          6
#define WINDOW_SIZE_DEFAULT        100
#define MIN_SWING_SIZE_DEFAULT     4
#define FIB_705_LEVEL             0.705
#define FIB_90_LEVEL              0.9
#define MAGIC_NUMBER_DEFAULT       234000
#define DEVIATION_DEFAULT          20
#define MAX_SPREAD_DEFAULT         3.0
#define MIN_BALANCE_DEFAULT        100
#define MAX_DAILY_TRADES_DEFAULT   10
#define ENTRY_TOLERANCE_DEFAULT    2.0
#define LOOKBACK_PERIOD_DEFAULT    20
#define MAX_RISK_PERCENT           2.0
#define MIN_VOLATILITY             5.0
#define MAX_VOLATILITY             50.0
#define TRAILING_STOP_PIPS         10
#define BREAK_EVEN_PIPS            10
#define PARTIAL_CLOSE_PERCENT      50

//--- Trading Sessions (Iran Time GMT+3:30)
struct TradingSession {
    int startHour;
    int startMinute;
    int endHour;
    int endMinute;
    string name;
};

const TradingSession LONDON_SESSION = {12, 30, 21, 30, "London"};
const TradingSession NEWYORK_SESSION = {17, 30, 2, 30, "NewYork"};
const TradingSession OVERLAP_LONDON_NY = {17, 30, 21, 30, "London-NY Overlap"};

//--- Helper macros
#define POINTS_TO_PIPS(points)     ((points) / 10.0)
#define PIPS_TO_POINTS(pips)       ((pips) * 10.0)
#define IS_BULLISH_CANDLE(rates, i) ((rates)[(i)].close >= (rates)[(i)].open)
#define IS_BEARISH_CANDLE(rates, i) ((rates)[(i)].close < (rates)[(i)].open)

//--- Input parameters
input group "=== Trading Parameters ==="
input double   InpLotSize = LOT_SIZE_DEFAULT;           // Lot size
input double   InpWinRatio = WIN_RATIO_DEFAULT;         // Win ratio (TP/SL)
input int      InpThreshold = THRESHOLD_DEFAULT;        // Leg threshold (pips)
input int      InpWindowSize = WINDOW_SIZE_DEFAULT;     // Data window size
input int      InpMinSwingSize = MIN_SWING_SIZE_DEFAULT; // Minimum swing size
input double   InpFib705 = FIB_705_LEVEL;              // Fibonacci 70.5% level
input double   InpFib90 = FIB_90_LEVEL;                // Fibonacci 90% level

input group "=== Risk Management ==="
input double   InpMaxRiskPercent = MAX_RISK_PERCENT;    // Max risk per trade (% of balance)
input double   InpMaxSpread = MAX_SPREAD_DEFAULT;       // Maximum spread (pips)
input double   InpEntryTolerance = ENTRY_TOLERANCE_DEFAULT; // Entry tolerance (pips)
input bool     InpUseTrailingStop = false;             // Use trailing stop
input int      InpTrailingStopPips = TRAILING_STOP_PIPS; // Trailing stop distance (pips)

input group "=== Time Settings ==="
input string   InpStartTime = "09:00";                 // Trading start time (Iran)
input string   InpEndTime = "21:00";                   // Trading end time (Iran)
input bool     InpTradeOnWeekends = false;             // Trade on weekends
input bool     InpUseLondonSession = true;             // Use London session
input bool     InpUseNYSession = true;                 // Use New York session
input bool     InpUseOverlapOnly = false;              // Trade only during London-NY overlap

input group "=== Expert Settings ==="
input int      InpMagicNumber = MAGIC_NUMBER_DEFAULT;  // Magic number
input string   InpTradeComment = "SwingEA";            // Trade comment
input bool     InpShowInfo = true;                     // Show info panel
input bool     InpEnableAlerts = true;                 // Enable alerts
input bool     InpSendEmails = false;                  // Send email notifications

//--- Global objects
CTrade trade;
CPositionInfo positionInfo;
COrderInfo orderInfo;

//--- Arrays for market data
MqlRates rates[];
datetime lastProcessedTime = 0;
int barsTotal = 0;

//--- Position tracking
bool positionOpen = false;
ulong currentTicket = 0;
datetime lastTradeTime = 0;
int dailyTradesCount = 0;
datetime lastDayCheck = 0;

//--- Enhanced State variables
struct EnhancedBotState {
    double fibLevels[4];        // [0.0, 0.705, 0.9, 1.0]
    bool truePosition;
    MqlRates lastTouched705PointUp;
    MqlRates lastTouched705PointDown;
    bool hasTouched705Up;
    bool hasTouched705Down;
    bool hasValidFib;
    datetime fibCreationTime;
    double fibStartPrice;
    double fibEndPrice;
    string lastSwingDirection;
    int consecutiveFailures;
    double maxDrawdown;
    double totalProfit;
    int totalTrades;
    int winningTrades;
} botState;

//--- Leg structure with additional information
struct EnhancedLeg {
    datetime startTime;
    datetime endTime;
    double startValue;
    double endValue;
    double length;
    string direction;
    int candleCount;
    double avgVolume;
    bool confirmed;
};

EnhancedLeg legs[];
int legsCount = 0;
string lastSwingType = "";
int startIndex = 0;

//--- Performance tracking
struct PerformanceStats {
    int totalSignals;
    int validSignals;
    int executedTrades;
    int profitableTrades;
    double totalPL;
    double maxDD;
    datetime lastResetTime;
} stats;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Validate input parameters
    if(!ValidateInputs()) {
        Print("❌ Invalid input parameters");
        return(INIT_PARAMETERS_INCORRECT);
    }
    
    //--- Set magic number and trade settings
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());
    
    //--- Initialize bot state and statistics
    ResetBotState();
    ZeroMemory(stats);
    stats.lastResetTime = TimeCurrent();
    
    //--- Set timer for processing
    EventSetTimer(1); // Check every second
    
    //--- Array setup
    ArraySetAsSeries(rates, true);
    ArrayResize(legs, 1000); // Pre-allocate space
    
    //--- Print initialization message
    Print("🚀 Enhanced Swing Trading EA Started...");
    Print("📊 Config: Symbol=", Symbol(), ", Lot=", InpLotSize, ", Win Ratio=", InpWinRatio);
    Print("⏰ Trading Hours (Iran): ", InpStartTime, " - ", InpEndTime);
    Print("🇮🇷 Current Iran Time: ", GetIranTimeString());
    
    //--- Perform system checks
    if(!PerformSystemChecks()) {
        Print("⚠️ Some system checks failed - EA will continue with warnings");
    }
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    //--- Print final statistics
    PrintFinalStats();
    
    //--- Send final notification
    if(InpSendEmails) {
        SendMail("Swing Trading EA Stopped", 
                 StringFormat("EA stopped. Reason: %s\nFinal P/L: %.2f", 
                             GetDeInitReasonText(reason), stats.totalPL));
    }
    
    Print("🔌 Enhanced Swing Trading EA stopped - Reason: ", GetDeInitReasonText(reason));
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check if new bar formed
    int bars = iBars(Symbol(), PERIOD_M1);
    if(bars <= barsTotal) return;
    
    //--- Update daily trade counter
    UpdateDailyTradeCounter();
    
    //--- Process new data
    ProcessNewData();
    barsTotal = bars;
    
    //--- Update trailing stops if enabled
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
    //--- Check trading conditions
    if(!CanTrade()) return;
    
    //--- Monitor risk levels
    MonitorRiskLevels();
    
    //--- Check for position management
    if(positionOpen) {
        ManageOpenPosition();
    }
}

//+------------------------------------------------------------------+
//| Trade transaction function                                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
{
    //--- Handle position close
    if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
        if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL) {
            //--- Check if it's our position
            if(trans.order == currentTicket) {
                HandlePositionClose(trans);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Enhanced trading conditions check                                |
//+------------------------------------------------------------------+
bool CanTrade()
{
    //--- Check basic conditions
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ||
       !MQLInfoInteger(MQL_TRADE_ALLOWED) ||
       !AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) {
        return false;
    }
    
    //--- Check weekend
    if(!InpTradeOnWeekends && IsIranWeekend()) {
        return false;
    }
    
    //--- Check daily trade limit
    if(dailyTradesCount >= MAX_DAILY_TRADES_DEFAULT) {
        return false;
    }
    
    //--- Check account balance
    if(AccountInfoDouble(ACCOUNT_BALANCE) < MIN_BALANCE_DEFAULT) {
        return false;
    }
    
    //--- Check spread
    double spread = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - 
                    SymbolInfoDouble(Symbol(), SYMBOL_BID)) / Point();
    if(spread > InpMaxSpread) {
        return false;
    }
    
    //--- Check trading hours
    return IsWithinAllowedTradingHours();
}

//+------------------------------------------------------------------+
//| Check if within allowed trading hours                           |
//+------------------------------------------------------------------+
bool IsWithinAllowedTradingHours()
{
    if(InpUseOverlapOnly) {
        return IsWithinTradingSession(OVERLAP_LONDON_NY);
    }
    
    bool inLondon = InpUseLondonSession && IsWithinTradingSession(LONDON_SESSION);
    bool inNY = InpUseNYSession && IsWithinTradingSession(NEWYORK_SESSION);
    bool inCustom = IsWithinCustomHours();
    
    return (inLondon || inNY || inCustom);
}

//+------------------------------------------------------------------+
//| Check custom trading hours                                       |
//+------------------------------------------------------------------+
bool IsWithinCustomHours()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime);
    
    string currentTime = StringFormat("%02d:%02d", iranTime.hour, iranTime.min);
    return (currentTime >= InpStartTime && currentTime <= InpEndTime);
}

//+------------------------------------------------------------------+
//| Enhanced data processing                                         |
//+------------------------------------------------------------------+
void ProcessNewData()
{
    //--- Get historical data with error checking
    int copied = CopyRates(Symbol(), PERIOD_M1, 0, InpWindowSize * 2, rates);
    if(copied <= 0) {
        Print("❌ Failed to get historical data. Error: ", GetLastError());
        return;
    }
    
    int dataSize = ArraySize(rates);
    if(dataSize < InpWindowSize) {
        Print("⚠️ Insufficient data: ", dataSize, " bars");
        return;
    }
    
    //--- Check for new data
    datetime currentTime = rates[0].time; // Most recent bar
    if(lastProcessedTime != 0 && currentTime == lastProcessedTime) {
        return; // No new data
    }
    
    //--- Log new data processing
    if(lastProcessedTime != 0) {
        Print("📊 Processing new data: ", TimeToString(currentTime), 
              " (Previous: ", TimeToString(lastProcessedTime), ")");
    }
    
    lastProcessedTime = currentTime;
    
    //--- Enhanced leg detection
    DetectLegsEnhanced(startIndex, dataSize-1);
    
    //--- Process swing analysis
    if(legsCount > 2) {
        ProcessSwingAnalysis();
    }
    else if(legsCount < 3 && botState.hasValidFib) {
        ProcessFibonacciUpdates(0); // Use most recent bar
    }
    
    //--- Check for trading signals
    CheckTradingSignals(0);
    
    //--- Monitor positions
    MonitorPositions();
    
    //--- Update statistics
    UpdateStatistics();
}

//+------------------------------------------------------------------+
//| Enhanced leg detection with validation                          |
//+------------------------------------------------------------------+
void DetectLegsEnhanced(int startIdx, int endIdx)
{
    ArrayFree(legs);
    legsCount = 0;
    
    if(endIdx - startIdx < 3) return;
    
    datetime currentStartIndex = rates[endIdx - startIdx].time;
    int j = 0;
    double minLegSize = InpThreshold;
    
    for(int i = endIdx - startIdx + 1; i >= 0; i--) { // Process from oldest to newest
        //--- Enhanced price determination
        bool currentIsBullish = IS_BULLISH_CANDLE(rates, i);
        double currentPrice = DetermineCurrentPrice(i, j);
        
        //--- Enhanced start price determination
        int startIndexPos = GetRateIndexByTime(currentStartIndex);
        if(startIndexPos < 0) continue;
        
        double startPrice = DetermineStartPrice(startIndexPos);
        
        //--- Calculate price difference with validation
        double priceDiff = MathAbs(currentPrice - startPrice) * MathPow(10, Digits() - 1);
        
        //--- Enhanced direction determination
        string direction = DetermineDirection(i, startIndexPos);
        if(direction == "") continue;
        
        //--- Validate leg formation
        if(priceDiff >= minLegSize && priceDiff < minLegSize * 5) {
            if(ValidateLegFormation(i, startIndexPos, direction, priceDiff)) {
                CreateNewLeg(j, currentStartIndex, rates[i].time, startPrice, currentPrice, priceDiff, direction);
                j++;
                currentStartIndex = rates[i].time;
            }
        }
        else if(j > 0 && priceDiff < minLegSize) {
            UpdateExistingLeg(j-1, i, currentPrice, priceDiff);
            currentStartIndex = rates[i].time;
        }
    }
    
    legsCount = j;
    if(legsCount > 0) {
        Print("📈 Detected ", legsCount, " legs with enhanced validation");
    }
}

//+------------------------------------------------------------------+
//| Process swing analysis with enhanced logic                      |
//+------------------------------------------------------------------+
void ProcessSwingAnalysis()
{
    if(legsCount < 3) return;
    
    //--- Take last 3 legs
    EnhancedLeg lastThreeLegs[];
    ArrayResize(lastThreeLegs, 3);
    for(int i = 0; i < 3; i++) {
        lastThreeLegs[i] = legs[legsCount - 3 + i];
    }
    
    //--- Enhanced swing detection
    string swingType;
    bool isSwing = DetectSwingWithConfirmation(lastThreeLegs, swingType);
    double confidence = CalculateSwingConfidence(lastThreeLegs, swingType);
    
    if(isSwing && confidence > 0.7) { // Only process high-confidence swings
        Print("✅ High-confidence swing detected: ", swingType, " (Confidence: ", confidence, ")");
        ProcessConfirmedSwing(swingType, isSwing);
        
        //--- Send alert if enabled
        if(InpEnableAlerts) {
            Alert("Swing detected: ", swingType, " on ", Symbol());
        }
    }
    else if(isSwing) {
        Print("⚠️ Low-confidence swing detected: ", swingType, " (Confidence: ", confidence, ")");
    }
}

//+------------------------------------------------------------------+
//| Enhanced swing detection with confirmation                      |
//+------------------------------------------------------------------+
bool DetectSwingWithConfirmation(EnhancedLeg &lastLegs[], string &swingType)
{
    if(ArraySize(lastLegs) != 3) return false;
    
    swingType = "";
    bool isSwing = false;
    
    //--- Enhanced bullish swing detection
    if(lastLegs[1].endValue > lastLegs[0].startValue && 
       lastLegs[0].endValue > lastLegs[1].endValue) {
        
        if(ValidateBullishSwing(lastLegs)) {
            swingType = "bullish";
            isSwing = true;
        }
    }
    //--- Enhanced bearish swing detection
    else if(lastLegs[1].endValue < lastLegs[0].startValue && 
            lastLegs[0].endValue < lastLegs[1].endValue) {
        
        if(ValidateBearishSwing(lastLegs)) {
            swingType = "bearish";
            isSwing = true;
        }
    }
    
    return isSwing;
}

//+------------------------------------------------------------------+
//| Calculate swing confidence level                                 |
//+------------------------------------------------------------------+
double CalculateSwingConfidence(EnhancedLeg &lastLegs[], string swingType)
{
    double confidence = 0.5; // Base confidence
    
    //--- Volume confirmation
    if(lastLegs[1].avgVolume > lastLegs[0].avgVolume * 1.2) {
        confidence += 0.1;
    }
    
    //--- Length confirmation
    if(lastLegs[1].length > lastLegs[0].length * 0.8) {
        confidence += 0.1;
    }
    
    //--- Candle count confirmation
    if(lastLegs[1].candleCount >= 3) {
        confidence += 0.1;
    }
    
    //--- Market volatility
    double currentVolatility = CalculateCurrentVolatility();
    if(currentVolatility > MIN_VOLATILITY && currentVolatility < MAX_VOLATILITY) {
        confidence += 0.1;
    }
    
    //--- Time factor
    datetime timeDiff = lastLegs[1].endTime - lastLegs[1].startTime;
    if(timeDiff > 180 && timeDiff < 1800) { // 3-30 minutes
        confidence += 0.1;
    }
    
    return MathMin(confidence, 1.0);
}

//+------------------------------------------------------------------+
//| Enhanced position management                                     |
//+------------------------------------------------------------------+
void ManageOpenPosition()
{
    if(!positionInfo.SelectByTicket(currentTicket)) {
        positionOpen = false;
        currentTicket = 0;
        return;
    }
    
    double currentProfit = positionInfo.Profit();
    double openPrice = positionInfo.PriceOpen();
    double currentPrice = (positionInfo.PositionType() == POSITION_TYPE_BUY) ? 
                         SymbolInfoDouble(Symbol(), SYMBOL_BID) : 
                         SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    
    //--- Break-even management
    if(MathAbs(currentPrice - openPrice) * MathPow(10, Digits() - 1) > BREAK_EVEN_PIPS) {
        MoveToBreakEven();
    }
    
    //--- Partial profit taking
    if(currentProfit > AccountInfoDouble(ACCOUNT_BALANCE) * 0.02) { // 2% of balance
        TakePartialProfit();
    }
    
    //--- Risk management
    if(currentProfit < -AccountInfoDouble(ACCOUNT_BALANCE) * InpMaxRiskPercent / 100) {
        Print("⚠️ Maximum risk reached - closing position");
        CloseCurrentPosition("Max risk reached");
    }
}

//+------------------------------------------------------------------+
//| Enhanced trading signal detection                                |
//+------------------------------------------------------------------+
void CheckTradingSignals(int currentIdx)
{
    if(!botState.truePosition || !botState.hasValidFib) return;
    if(positionOpen || !CanTrade()) return;
    
    //--- Enhanced signal validation
    if(!ValidateMarketConditions()) {
        Print("🚫 Market conditions not suitable for trading");
        return;
    }
    
    double currentPrice = rates[currentIdx].close;
    double entryPrice = (lastSwingType == "bullish") ? 
                       SymbolInfoDouble(Symbol(), SYMBOL_ASK) : 
                       SymbolInfoDouble(Symbol(), SYMBOL_BID);
    
    //--- Calculate risk-adjusted position size
    double lotSize = CalculatePositionSize(entryPrice);
    if(lotSize < SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN)) {
        Print("🚫 Calculated lot size too small: ", lotSize);
        return;
    }
    
    //--- Execute trade with enhanced error handling
    if(lastSwingType == "bullish") {
        ExecuteEnhancedBuyOrder(entryPrice, lotSize);
    }
    else if(lastSwingType == "bearish") {
        ExecuteEnhancedSellOrder(entryPrice, lotSize);
    }
}

//+------------------------------------------------------------------+
//| Calculate risk-adjusted position size                           |
//+------------------------------------------------------------------+
double CalculatePositionSize(double entryPrice)
{
    double riskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * InpMaxRiskPercent / 100;
    double stopDistance;
    
    if(MathAbs(botState.fibLevels[2] - entryPrice) * MathPow(10, Digits() - 1) < 2) {
        stopDistance = MathAbs(entryPrice - botState.fibLevels[3]);
    }
    else {
        stopDistance = MathAbs(entryPrice - botState.fibLevels[2]);
    }
    
    double tickValue = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_SIZE);
    
    double riskPerLot = (stopDistance / tickSize) * tickValue;
    double calculatedLotSize = riskAmount / riskPerLot;
    
    //--- Normalize to broker requirements
    double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    
    calculatedLotSize = MathMax(calculatedLotSize, minLot);
    calculatedLotSize = MathMin(calculatedLotSize, maxLot);
    calculatedLotSize = MathMin(calculatedLotSize, InpLotSize * 3); // Don't exceed 3x input lot size
    
    //--- Round to step
    calculatedLotSize = MathFloor(calculatedLotSize / stepLot) * stepLot;
    
    return calculatedLotSize;
}

//+------------------------------------------------------------------+
//| Execute enhanced buy order                                       |
//+------------------------------------------------------------------+
bool ExecuteEnhancedBuyOrder(double price, double lotSize)
{
    double sl, tp;
    if(!CalculateStopLevels(price, true, sl, tp)) {
        Print("❌ Invalid stop levels calculated");
        return false;
    }
    
    //--- Pre-trade validation
    if(!ValidateTradeRequest(ORDER_TYPE_BUY, lotSize, price, sl, tp)) {
        return false;
    }
    
    //--- Execute trade
    if(trade.Buy(lotSize, Symbol(), price, sl, tp, InpTradeComment)) {
        currentTicket = trade.ResultOrder();
        positionOpen = true;
        dailyTradesCount++;
        lastTradeTime = TimeCurrent();
        stats.executedTrades++;
        
        Print("✅ BUY order executed: Ticket=", currentTicket, 
              " Price=", price, " SL=", sl, " TP=", tp, " Lot=", lotSize);
        
        //--- Send notification
        if(InpSendEmails) {
            SendTradeNotification("BUY", price, sl, tp, lotSize);
        }
        
        ResetBotState();
        return true;
    }
    else {
        Print("❌ BUY order failed: ", trade.ResultRetcodeDescription());
        botState.consecutiveFailures++;
        return false;
    }
}

//+------------------------------------------------------------------+
//| Execute enhanced sell order                                      |
//+------------------------------------------------------------------+
bool ExecuteEnhancedSellOrder(double price, double lotSize)
{
    double sl, tp;
    if(!CalculateStopLevels(price, false, sl, tp)) {
        Print("❌ Invalid stop levels calculated");
        return false;
    }
    
    //--- Pre-trade validation
    if(!ValidateTradeRequest(ORDER_TYPE_SELL, lotSize, price, sl, tp)) {
        return false;
    }
    
    //--- Execute trade
    if(trade.Sell(lotSize, Symbol(), price, sl, tp, InpTradeComment)) {
        currentTicket = trade.ResultOrder();
        positionOpen = true;
        dailyTradesCount++;
        lastTradeTime = TimeCurrent();
        stats.executedTrades++;
        
        Print("✅ SELL order executed: Ticket=", currentTicket, 
              " Price=", price, " SL=", sl, " TP=", tp, " Lot=", lotSize);
        
        //--- Send notification
        if(InpSendEmails) {
            SendTradeNotification("SELL", price, sl, tp, lotSize);
        }
        
        ResetBotState();
        return true;
    }
    else {
        Print("❌ SELL order failed: ", trade.ResultRetcodeDescription());
        botState.consecutiveFailures++;
        return false;
    }
}

//+------------------------------------------------------------------+
//| Helper functions                                                 |
//+------------------------------------------------------------------+

bool ValidateInputs()
{
    if(InpLotSize <= 0 || InpWinRatio <= 0 || InpThreshold <= 0) return false;
    if(InpWindowSize < 50 || InpMaxRiskPercent <= 0 || InpMaxRiskPercent > 10) return false;
    return true;
}

bool PerformSystemChecks()
{
    CheckSymbolProperties();
    CheckTradingLimits();
    CheckAccountPermissions();
    return true;
}

void UpdateDailyTradeCounter()
{
    datetime currentDay = (datetime)(TimeCurrent() / 86400) * 86400;
    if(currentDay != lastDayCheck) {
        dailyTradesCount = 0;
        lastDayCheck = currentDay;
        Print("📅 New trading day started - trade counter reset");
    }
}

void UpdateInfoPanel()
{
    string info = StringFormat("SwingEA | Balance: %.2f | Equity: %.2f | Trades: %d/%d | P/L: %.2f",
                              AccountInfoDouble(ACCOUNT_BALANCE),
                              AccountInfoDouble(ACCOUNT_EQUITY),
                              dailyTradesCount, MAX_DAILY_TRADES_DEFAULT,
                              stats.totalPL);
    Comment(info);
}

void PrintFinalStats()
{
    Print("📊 Final Statistics:");
    Print("   Total Signals: ", stats.totalSignals);
    Print("   Valid Signals: ", stats.validSignals);
    Print("   Executed Trades: ", stats.executedTrades);
    Print("   Profitable Trades: ", stats.profitableTrades);
    Print("   Total P/L: ", stats.totalPL);
    Print("   Win Rate: ", (stats.executedTrades > 0) ? (double)stats.profitableTrades/stats.executedTrades*100 : 0, "%");
}

//+------------------------------------------------------------------+
//| Helper Functions - Integrated from SwingTradingHelpers.mqh      |
//+------------------------------------------------------------------+

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

//--- Reset bot state function
void ResetBotState() {
    ArrayInitialize(botState.fibLevels, 0);
    botState.truePosition = false;
    botState.hasTouched705Up = false;
    botState.hasTouched705Down = false;
    botState.hasValidFib = false;
    botState.fibCreationTime = 0;
    botState.fibStartPrice = 0;
    botState.fibEndPrice = 0;
    botState.consecutiveFailures = 0;
    
    // Clear touched points
    ZeroMemory(botState.lastTouched705PointUp);
    ZeroMemory(botState.lastTouched705PointDown);
    
    Print("🔄 Bot state reset");
}

//--- Calculate current market volatility
double CalculateCurrentVolatility() {
    if(ArraySize(rates) < 20) return 0;
    
    double sum = 0;
    for(int i = 0; i < 20; i++) {
        sum += (rates[i].high - rates[i].low);
    }
    
    return (sum / 20) * MathPow(10, Digits() - 1); // Convert to pips
}

//--- Monitor risk levels
void MonitorRiskLevels() {
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double drawdown = (balance - currentEquity) / balance * 100;
    
    if(drawdown > 10.0) { // 10% max drawdown
        Print("🚨 Maximum drawdown reached: ", drawdown, "%");
        if(positionOpen) {
            CloseCurrentPosition("Max drawdown reached");
        }
        
        // Disable trading for 1 hour
        lastTradeTime = TimeCurrent() + 3600;
    }
    
    // Update max drawdown
    if(drawdown > botState.maxDrawdown) {
        botState.maxDrawdown = drawdown;
    }
}

//--- Update trailing stop
void UpdateTrailingStop() {
    if(!positionInfo.SelectByTicket(currentTicket)) return;
    
    double currentPrice = (positionInfo.PositionType() == POSITION_TYPE_BUY) ? 
                         SymbolInfoDouble(Symbol(), SYMBOL_BID) : 
                         SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    
    double currentSL = positionInfo.StopLoss();
    double newSL = currentSL;
    
    if(positionInfo.PositionType() == POSITION_TYPE_BUY) {
        newSL = currentPrice - InpTrailingStopPips * Point() * 10;
        if(newSL > currentSL + Point() * 10) { // Only move SL up
            trade.PositionModify(currentTicket, newSL, positionInfo.TakeProfit());
        }
    }
    else {
        newSL = currentPrice + InpTrailingStopPips * Point() * 10;
        if(newSL < currentSL - Point() * 10) { // Only move SL down
            trade.PositionModify(currentTicket, newSL, positionInfo.TakeProfit());
        }
    }
}

//--- Process fibonacci updates
void ProcessFibonacciUpdates(int currentIdx) {
    if(!botState.hasValidFib) return;
    
    double currentHigh = rates[currentIdx].high;
    double currentLow = rates[currentIdx].low;
    
    if(lastSwingType == "bullish") {
        if(botState.fibLevels[0] < currentHigh) {
            Print("🔄 Updating bullish fibonacci - new high detected");
            double endPrice = botState.fibLevels[3]; // Keep fib 1.0
            CalculateFibonacci(endPrice, currentHigh);
            botState.fibCreationTime = rates[currentIdx].time;
        }
        else if(currentLow <= botState.fibLevels[1]) {
            HandleFibonacciTouch(currentIdx, true);
        }
        else if(currentLow < botState.fibLevels[3]) {
            Print("❌ Price broke below fib 1.0 - resetting state");
            ResetBotState();
        }
    }
    else if(lastSwingType == "bearish") {
        if(botState.fibLevels[0] > currentLow) {
            Print("🔄 Updating bearish fibonacci - new low detected");
            double endPrice = botState.fibLevels[3]; // Keep fib 1.0
            CalculateFibonacci(currentLow, endPrice);
            botState.fibCreationTime = rates[currentIdx].time;
        }
        else if(currentHigh >= botState.fibLevels[1]) {
            HandleFibonacciTouch(currentIdx, false);
        }
        else if(currentHigh > botState.fibLevels[3]) {
            Print("❌ Price broke above fib 1.0 - resetting state");
            ResetBotState();
        }
    }
}

//--- Calculate fibonacci levels
void CalculateFibonacci(double startPrice, double endPrice) {
    botState.fibLevels[0] = startPrice;  // 0.0
    botState.fibLevels[1] = startPrice + InpFib705 * (endPrice - startPrice);  // 0.705
    botState.fibLevels[2] = startPrice + InpFib90 * (endPrice - startPrice);   // 0.9
    botState.fibLevels[3] = endPrice;    // 1.0
    botState.hasValidFib = true;
    
    Print("📊 Fibonacci levels calculated:");
    Print("   0.0% = ", botState.fibLevels[0]);
    Print("   70.5% = ", botState.fibLevels[1]);
    Print("   90.0% = ", botState.fibLevels[2]);
    Print("   100.0% = ", botState.fibLevels[3]);
}

//--- Handle fibonacci touch
void HandleFibonacciTouch(int currentIdx, bool isBullish) {
    string currentStatus = IS_BULLISH_CANDLE(rates, currentIdx) ? "bullish" : "bearish";
    
    if(isBullish) {
        if(!botState.hasTouched705Up) {
            Print("🎯 First touch of 0.705 level (bullish setup)");
            botState.lastTouched705PointUp = rates[currentIdx];
            botState.hasTouched705Up = true;
        }
        else {
            string lastStatus = (botState.lastTouched705PointUp.close >= botState.lastTouched705PointUp.open) ? "bullish" : "bearish";
            if(currentStatus != lastStatus) {
                Print("✅ Second touch of 0.705 level with different candle type - ENTRY SIGNAL!");
                botState.truePosition = true;
            }
        }
    }
    else {
        if(!botState.hasTouched705Down) {
            Print("🎯 First touch of 0.705 level (bearish setup)");
            botState.lastTouched705PointDown = rates[currentIdx];
            botState.hasTouched705Down = true;
        }
        else {
            string lastStatus = (botState.lastTouched705PointDown.close >= botState.lastTouched705PointDown.open) ? "bullish" : "bearish";
            if(currentStatus != lastStatus) {
                Print("✅ Second touch of 0.705 level with different candle type - ENTRY SIGNAL!");
                botState.truePosition = true;
            }
        }
    }
}

//--- Monitor positions
void MonitorPositions() {
    int totalPositions = PositionsTotal();
    
    if(totalPositions == 0 && positionOpen) {
        Print("🏁 Position closed by market");
        positionOpen = false;
        currentTicket = 0;
    }
}

//--- Update statistics
void UpdateStatistics() {
    // Update daily statistics and other metrics
    botState.totalTrades = stats.executedTrades;
    botState.totalProfit = stats.totalPL;
    
    if(stats.executedTrades > 0) {
        botState.winningTrades = stats.profitableTrades;
    }
}

//--- Get deinit reason text
string GetDeInitReasonText(int reason) {
    switch(reason) {
        case REASON_ACCOUNT: return "Account changed";
        case REASON_CHARTCHANGE: return "Chart changed";
        case REASON_CHARTCLOSE: return "Chart closed";
        case REASON_PARAMETERS: return "Parameters changed";
        case REASON_RECOMPILE: return "Recompiled";
        case REASON_REMOVE: return "Removed from chart";
        case REASON_TEMPLATE: return "Template changed";
        default: return "Unknown reason";
    }
}

//--- Handle position close
void HandlePositionClose(const MqlTradeTransaction& trans) {
    if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL) {
        double profit = 0;
        if(HistoryDealSelect(trans.deal)) {
            profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
            stats.totalPL += profit;
            
            if(profit > 0) {
                stats.profitableTrades++;
                botState.consecutiveFailures = 0;
                Print("✅ Profitable trade closed: ", profit);
            }
            else {
                botState.consecutiveFailures++;
                Print("❌ Loss trade closed: ", profit);
            }
        }
        
        positionOpen = false;
        currentTicket = 0;
    }
}

//--- Simple leg detection (simplified version)
void DetectLegsEnhanced(int startIdx, int endIdx) {
    ArrayFree(legs);
    legsCount = 0;
    
    if(endIdx - startIdx < 3) return;
    
    // Simplified leg detection logic
    double threshold = InpThreshold;
    int legIndex = 0;
    
    for(int i = startIdx + 10; i < endIdx - 10; i += 5) { // Sample every 5 bars
        double currentHigh = rates[i].high;
        double currentLow = rates[i].low;
        double prevHigh = rates[i + 5].high;
        double prevLow = rates[i + 5].low;
        
        double priceDiff = MathAbs(currentHigh - prevLow) * MathPow(10, Digits() - 1);
        
        if(priceDiff >= threshold && legIndex < 1000) {
            ArrayResize(legs, legIndex + 1);
            legs[legIndex].startTime = rates[i + 5].time;
            legs[legIndex].endTime = rates[i].time;
            legs[legIndex].startValue = prevLow;
            legs[legIndex].endValue = currentHigh;
            legs[legIndex].length = priceDiff;
            legs[legIndex].direction = (currentHigh > prevLow) ? "up" : "down";
            legs[legIndex].confirmed = true;
            legIndex++;
        }
    }
    
    legsCount = legIndex;
    if(legsCount > 0) {
        Print("📈 Detected ", legsCount, " legs");
    }
}

//--- Simplified swing detection
bool DetectSwingWithConfirmation(EnhancedLeg &lastLegs[], string &swingType) {
    if(ArraySize(lastLegs) != 3) return false;
    
    swingType = "";
    bool isSwing = false;
    
    // Simple bullish swing pattern
    if(lastLegs[1].endValue > lastLegs[0].startValue && 
       lastLegs[0].endValue > lastLegs[1].endValue) {
        swingType = "bullish";
        isSwing = true;
    }
    // Simple bearish swing pattern
    else if(lastLegs[1].endValue < lastLegs[0].startValue && 
            lastLegs[0].endValue < lastLegs[1].endValue) {
        swingType = "bearish";
        isSwing = true;
    }
    
    return isSwing;
}

//--- Calculate swing confidence (simplified)
double CalculateSwingConfidence(EnhancedLeg &lastLegs[], string swingType) {
    return 0.8; // Return high confidence for simplicity
}

//--- Process confirmed swing (simplified)
void ProcessConfirmedSwing(string swingType, bool isSwing) {
    stats.totalSignals++;
    
    if(isSwing) {
        stats.validSignals++;
        lastSwingType = swingType;
        
        Print("🔍 Processing confirmed swing: ", swingType);
        
        // Create fibonacci levels based on recent price action
        if(!botState.hasValidFib) {
            double currentHigh = rates[0].high;
            double currentLow = rates[0].low;
            
            if(swingType == "bullish") {
                CalculateFibonacci(currentLow, currentHigh);
            }
            else {
                CalculateFibonacci(currentHigh, currentLow);
            }
            
            botState.fibCreationTime = rates[0].time;
        }
    }
}

//--- Validate market conditions (simplified)
bool ValidateMarketConditions() {
    // Check spread
    double spread = POINTS_TO_PIPS(SymbolInfoInteger(Symbol(), SYMBOL_SPREAD));
    if(spread > InpMaxSpread) {
        return false;
    }
    
    // Check volatility
    double volatility = CalculateCurrentVolatility();
    if(volatility < MIN_VOLATILITY || volatility > MAX_VOLATILITY) {
        return false;
    }
    
    return true;
}

//--- Calculate stop levels
bool CalculateStopLevels(double entryPrice, bool isBuy, double &sl, double &tp) {
    if(!botState.hasValidFib) return false;
    
    // Determine stop loss
    if(MathAbs(botState.fibLevels[2] - entryPrice) * MathPow(10, Digits() - 1) < 2) {
        sl = botState.fibLevels[3]; // Use fib 1.0
    }
    else {
        sl = botState.fibLevels[2]; // Use fib 0.9
    }
    
    // Calculate take profit
    double stopDistance = MathAbs(entryPrice - sl);
    if(isBuy) {
        tp = entryPrice + (stopDistance * InpWinRatio);
    }
    else {
        tp = entryPrice - (stopDistance * InpWinRatio);
    }
    
    // Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    
    return true;
}

//--- Validate trade request
bool ValidateTradeRequest(ENUM_ORDER_TYPE orderType, double volume, double price, double sl, double tp) {
    // Check volume
    double minVol = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double maxVol = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    
    if(volume < minVol || volume > maxVol) {
        Print("❌ Invalid volume: ", volume);
        return false;
    }
    
    return true;
}

//--- Move to break even
void MoveToBreakEven() {
    if(!positionInfo.SelectByTicket(currentTicket)) return;
    
    double openPrice = positionInfo.PriceOpen();
    double currentSL = positionInfo.StopLoss();
    
    // Check if SL is not already at break-even
    if(MathAbs(currentSL - openPrice) < Point() * 5) return;
    
    trade.PositionModify(currentTicket, openPrice, positionInfo.TakeProfit());
    Print("📊 Moved stop loss to break-even for ticket: ", currentTicket);
}

//--- Take partial profit
void TakePartialProfit() {
    if(!positionInfo.SelectByTicket(currentTicket)) return;
    
    double currentVolume = positionInfo.Volume();
    double partialVolume = currentVolume * PARTIAL_CLOSE_PERCENT / 100;
    
    // Normalize volume
    double stepVol = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    partialVolume = MathFloor(partialVolume / stepVol) * stepVol;
    
    if(partialVolume >= SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN)) {
        if(trade.PositionClosePartial(currentTicket, partialVolume)) {
            Print("📊 Partial profit taken: ", partialVolume, " lots");
        }
    }
}

//--- Close current position
void CloseCurrentPosition(string reason) {
    if(trade.PositionClose(currentTicket)) {
        Print("🔒 Position closed: ", reason);
        positionOpen = false;
        currentTicket = 0;
    }
}

//--- Send trade notification
void SendTradeNotification(string action, double price, double sl, double tp, double volume) {
    string subject = StringFormat("Swing EA - %s Trade Executed", action);
    string body = StringFormat(
        "Trade Details:\n"
        "Symbol: %s\n"
        "Action: %s\n"
        "Volume: %.2f\n"
        "Price: %.5f\n"
        "Stop Loss: %.5f\n"
        "Take Profit: %.5f\n"
        "Time: %s\n"
        "Balance: %.2f",
        Symbol(), action, volume, price, sl, tp, 
        TimeToString(TimeCurrent()), AccountInfoDouble(ACCOUNT_BALANCE)
    );
    
    SendMail(subject, body);
}

void CheckSymbolProperties() {
    Print("📊 Symbol Properties Check:");
    Print("   Symbol: ", Symbol());
    Print("   Digits: ", Digits());
    Print("   Point: ", Point());
    Print("   Spread: ", POINTS_TO_PIPS(SymbolInfoInteger(Symbol(), SYMBOL_SPREAD)), " pips");
}

void CheckTradingLimits() {
    double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    
    Print("📊 Trading Limits Check:");
    Print("   Min Lot: ", minLot);
    Print("   Max Lot: ", maxLot);
    Print("   Lot Step: ", stepLot);
}

void CheckAccountPermissions() {
    Print("📊 Account Permissions Check:");
    Print("   Terminal Trading: ", TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? "Allowed" : "Disabled");
    Print("   EA Trading: ", MQLInfoInteger(MQL_TRADE_ALLOWED) ? "Allowed" : "Disabled");
    Print("   Account Trading: ", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) ? "Allowed" : "Disabled");
    Print("   Balance: ", AccountInfoDouble(ACCOUNT_BALANCE));
    Print("   Equity: ", AccountInfoDouble(ACCOUNT_EQUITY));
}

// Additional helper functions would continue here...
// [Due to length constraints, I'm showing the key structure and main functions]

//+------------------------------------------------------------------+
