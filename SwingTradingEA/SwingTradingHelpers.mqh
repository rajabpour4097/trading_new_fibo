//+------------------------------------------------------------------+
//|                                           SwingTradingHelpers.mqh |
//|                    Helper functions for Swing Trading EA        |
//|                                                                  |
//+------------------------------------------------------------------+

#ifndef SWING_TRADING_HELPERS_MQH
#define SWING_TRADING_HELPERS_MQH

//+------------------------------------------------------------------+
//| Additional helper functions for Enhanced EA                     |
//+------------------------------------------------------------------+

//--- Remaining helper functions for SwingTradingEA_Enhanced.mq5

double DetermineCurrentPrice(int index, int legCount)
{
    bool currentIsBullish = IS_BULLISH_CANDLE(rates, index);
    
    if(legCount > 0 && legs[legCount-1].direction == "up" && 
       rates[index].high >= rates[index+1].high) {
        return rates[index].high;
    }
    else if(legCount > 0 && legs[legCount-1].direction == "down" && 
            rates[index].low <= rates[index+1].low) {
        return rates[index].low;
    }
    else {
        return currentIsBullish ? rates[index].high : rates[index].low;
    }
}

double DetermineStartPrice(int startIndexPos)
{
    bool startIsBullish = IS_BULLISH_CANDLE(rates, startIndexPos);
    return startIsBullish ? rates[startIndexPos].high : rates[startIndexPos].low;
}

string DetermineDirection(int currentIdx, int startIdx)
{
    if(rates[currentIdx].close >= rates[startIdx].close || 
       (rates[currentIdx].high > rates[currentIdx+1].high && 
        rates[currentIdx].close >= rates[currentIdx+1].close)) {
        return "up";
    }
    else if(rates[currentIdx].close < rates[startIdx].close || 
            (rates[currentIdx].low < rates[currentIdx+1].low && 
             rates[currentIdx].close < rates[currentIdx+1].close)) {
        return "down";
    }
    return "";
}

bool ValidateLegFormation(int currentIdx, int startIdx, string direction, double priceDiff)
{
    //--- Check minimum time duration
    int timeBars = MathAbs(currentIdx - startIdx);
    if(timeBars < 3) return false;
    
    //--- Check price movement consistency
    int confirmationBars = 0;
    if(direction == "up") {
        for(int i = startIdx; i >= currentIdx; i--) {
            if(rates[i].close > rates[i].open) confirmationBars++;
        }
    }
    else {
        for(int i = startIdx; i >= currentIdx; i--) {
            if(rates[i].close < rates[i].open) confirmationBars++;
        }
    }
    
    return (confirmationBars >= timeBars * 0.4); // At least 40% confirmation
}

void CreateNewLeg(int index, datetime startTime, datetime endTime, 
                 double startValue, double endValue, double length, string direction)
{
    if(index >= ArraySize(legs)) {
        ArrayResize(legs, index + 100);
    }
    
    legs[index].startTime = startTime;
    legs[index].endTime = endTime;
    legs[index].startValue = startValue;
    legs[index].endValue = endValue;
    legs[index].length = length;
    legs[index].direction = direction;
    legs[index].confirmed = true;
    
    //--- Calculate additional metrics
    int startIdx = GetRateIndexByTime(startTime);
    int endIdx = GetRateIndexByTime(endTime);
    
    if(startIdx >= 0 && endIdx >= 0) {
        legs[index].candleCount = MathAbs(startIdx - endIdx) + 1;
        legs[index].avgVolume = CalculateAverageVolume(endIdx, startIdx);
    }
}

void UpdateExistingLeg(int legIndex, int currentIdx, double currentPrice, double priceDiff)
{
    if(legIndex < 0 || legIndex >= legsCount) return;
    
    legs[legIndex].endTime = rates[currentIdx].time;
    legs[legIndex].endValue = currentPrice;
    legs[legIndex].length = priceDiff;
}

int GetRateIndexByTime(datetime time)
{
    for(int i = 0; i < ArraySize(rates); i++) {
        if(rates[i].time == time) return i;
    }
    return -1;
}

double CalculateAverageVolume(int startIdx, int endIdx)
{
    if(startIdx < 0 || endIdx < 0 || startIdx >= ArraySize(rates) || endIdx >= ArraySize(rates)) 
        return 0;
    
    long totalVolume = 0;
    int count = 0;
    
    for(int i = endIdx; i <= startIdx; i++) {
        totalVolume += rates[i].tick_volume;
        count++;
    }
    
    return (count > 0) ? (double)totalVolume / count : 0;
}

bool ValidateBullishSwing(EnhancedLeg &lastLegs[])
{
    //--- Enhanced validation for bullish swing
    int startIdx = GetRateIndexByTime(lastLegs[1].startTime);
    int endIdx = GetRateIndexByTime(lastLegs[1].endTime);
    
    if(startIdx < 0 || endIdx < 0) return false;
    
    int trueCandles = 0;
    bool firstCandle = false;
    double lastCandleClose = 0;
    
    for(int k = startIdx; k >= endIdx; k--) {
        if(IS_BEARISH_CANDLE(rates, k)) {
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
    
    return (trueCandles >= 3);
}

bool ValidateBearishSwing(EnhancedLeg &lastLegs[])
{
    //--- Enhanced validation for bearish swing
    int startIdx = GetRateIndexByTime(lastLegs[1].startTime);
    int endIdx = GetRateIndexByTime(lastLegs[1].endTime);
    
    if(startIdx < 0 || endIdx < 0) return false;
    
    int trueCandles = 0;
    bool firstCandle = false;
    double lastCandleClose = 0;
    
    for(int k = startIdx; k >= endIdx; k--) {
        if(IS_BULLISH_CANDLE(rates, k)) {
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
    
    return (trueCandles >= 3);
}

void ProcessConfirmedSwing(string swingType, bool isSwing)
{
    stats.totalSignals++;
    
    if(isSwing || botState.hasValidFib) {
        stats.validSignals++;
        
        Print("🔍 Processing confirmed swing: ", swingType);
        
        double currentClose = rates[0].close;
        double currentHigh = rates[0].high;
        double currentLow = rates[0].low;
        
        //--- Phase 1: Initial swing detection
        if(isSwing && !botState.hasValidFib) {
            ProcessInitialSwing(swingType, currentHigh, currentLow, currentClose);
        }
        //--- Phase 2: Update existing fibonacci in same direction
        else if(isSwing && botState.hasValidFib && lastSwingType == swingType) {
            ProcessSameDirectionSwing(swingType, currentHigh, currentLow);
        }
        //--- Phase 3: Opposite swing handling
        else if(isSwing && botState.hasValidFib && lastSwingType != swingType) {
            ProcessOppositeDirectionSwing(swingType, currentHigh, currentLow);
        }
        //--- Phase 4: Update without swing
        else if(!isSwing && botState.hasValidFib) {
            ProcessFibonacciUpdates(0);
        }
    }
}

void ProcessInitialSwing(string swingType, double currentHigh, double currentLow, double currentClose)
{
    Print("🎯 Initial swing detection: ", swingType);
    lastSwingType = swingType;
    
    if(swingType == "bullish") {
        if(currentClose >= legs[legsCount-3].endValue) {
            double startPrice = currentHigh;
            double endPrice = legs[legsCount-2].endValue;
            
            if(currentHigh >= legs[legsCount-2].endValue) {
                Print("✅ Creating bullish fibonacci levels");
                CalculateFibonacci(endPrice, startPrice);
                botState.fibCreationTime = rates[0].time;
                botState.fibStartPrice = endPrice;
                botState.fibEndPrice = startPrice;
            }
        }
    }
    else if(swingType == "bearish") {
        if(currentClose <= legs[legsCount-3].endValue) {
            double startPrice = currentLow;
            double endPrice = legs[legsCount-2].endValue;
            
            if(currentLow <= legs[legsCount-2].endValue) {
                Print("✅ Creating bearish fibonacci levels");
                CalculateFibonacci(startPrice, endPrice);
                botState.fibCreationTime = rates[0].time;
                botState.fibStartPrice = startPrice;
                botState.fibEndPrice = endPrice;
            }
        }
    }
}

void ProcessSameDirectionSwing(string swingType, double currentHigh, double currentLow)
{
    Print("🔄 Updating fibonacci in same direction: ", swingType);
    
    if(swingType == "bullish") {
        if(currentHigh >= legs[legsCount-2].endValue) {
            double startPrice = currentHigh;
            double endPrice = legs[legsCount-2].endValue;
            CalculateFibonacci(endPrice, startPrice);
            botState.fibCreationTime = rates[0].time;
        }
        else if(currentLow <= botState.fibLevels[1]) { // 0.705 level
            HandleFibonacciTouch(0, true);
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
            botState.fibCreationTime = rates[0].time;
        }
        else if(currentHigh >= botState.fibLevels[1]) { // 0.705 level
            HandleFibonacciTouch(0, false);
        }
        else if(currentHigh > botState.fibLevels[3]) { // Above 1.0 level
            Print("❌ Price broke above fib 1.0 - resetting");
            ResetBotState();
        }
    }
}

void ProcessOppositeDirectionSwing(string swingType, double currentHigh, double currentLow)
{
    Print("⚠️ Opposite swing detected - checking fib 1.0 violation");
    
    if(lastSwingType == "bullish" && swingType == "bearish") {
        if(currentLow < botState.fibLevels[3]) {
            Print("❌ Bearish swing violated fib 1.0 - resetting");
            ResetBotState();
        }
        else {
            Print("📊 Bearish swing within fib range - ignoring");
        }
    }
    else if(lastSwingType == "bearish" && swingType == "bullish") {
        if(currentHigh > botState.fibLevels[3]) {
            Print("❌ Bullish swing violated fib 1.0 - resetting");
            ResetBotState();
        }
        else {
            Print("📊 Bullish swing within fib range - ignoring");
        }
    }
}

double CalculateCurrentVolatility()
{
    if(ArraySize(rates) < 20) return 0;
    
    double sum = 0;
    for(int i = 0; i < 20; i++) {
        sum += (rates[i].high - rates[i].low);
    }
    
    return (sum / 20) * MathPow(10, Digits() - 1); // Convert to pips
}

bool ValidateMarketConditions()
{
    //--- Check spread
    double spread = POINTS_TO_PIPS(SymbolInfoInteger(Symbol(), SYMBOL_SPREAD));
    if(spread > InpMaxSpread) {
        Print("🚫 Spread too high: ", spread, " pips");
        return false;
    }
    
    //--- Check volatility
    double volatility = CalculateCurrentVolatility();
    if(volatility < MIN_VOLATILITY || volatility > MAX_VOLATILITY) {
        Print("🚫 Volatility outside range: ", volatility, " pips");
        return false;
    }
    
    //--- Check time since last fibonacci creation
    if(botState.hasValidFib && TimeCurrent() - botState.fibCreationTime < 300) { // 5 minutes
        Print("🚫 Too soon after fibonacci creation");
        return false;
    }
    
    //--- Check consecutive failures
    if(botState.consecutiveFailures >= 3) {
        Print("🚫 Too many consecutive failures");
        return false;
    }
    
    return true;
}

bool CalculateStopLevels(double entryPrice, bool isBuy, double &sl, double &tp)
{
    if(!botState.hasValidFib) return false;
    
    //--- Determine stop loss
    if(MathAbs(botState.fibLevels[2] - entryPrice) * MathPow(10, Digits() - 1) < 2) {
        sl = botState.fibLevels[3]; // Use fib 1.0
    }
    else {
        sl = botState.fibLevels[2]; // Use fib 0.9
    }
    
    //--- Calculate take profit
    double stopDistance = MathAbs(entryPrice - sl);
    if(isBuy) {
        tp = entryPrice + (stopDistance * InpWinRatio);
    }
    else {
        tp = entryPrice - (stopDistance * InpWinRatio);
    }
    
    //--- Normalize prices
    sl = NormalizeDouble(sl, Digits());
    tp = NormalizeDouble(tp, Digits());
    entryPrice = NormalizeDouble(entryPrice, Digits());
    
    //--- Validate stop levels
    double minStopLevel = SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL) * Point();
    
    if(isBuy) {
        if(entryPrice - sl < minStopLevel || tp - entryPrice < minStopLevel) {
            Print("❌ Stop levels too close to entry price");
            return false;
        }
    }
    else {
        if(sl - entryPrice < minStopLevel || entryPrice - tp < minStopLevel) {
            Print("❌ Stop levels too close to entry price");
            return false;
        }
    }
    
    return true;
}

bool ValidateTradeRequest(ENUM_ORDER_TYPE orderType, double volume, double price, double sl, double tp)
{
    //--- Check volume
    double minVol = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double maxVol = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    
    if(volume < minVol || volume > maxVol) {
        Print("❌ Invalid volume: ", volume);
        return false;
    }
    
    //--- Check margin requirements
    double margin = 0;
    if(!OrderCalcMargin(orderType, Symbol(), volume, price, margin)) {
        Print("❌ Cannot calculate margin requirement");
        return false;
    }
    
    double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    if(margin > freeMargin * 0.8) { // Use max 80% of free margin
        Print("❌ Insufficient margin. Required: ", margin, " Available: ", freeMargin);
        return false;
    }
    
    //--- Check profit calculation
    double profit = 0;
    if(!OrderCalcProfit(orderType, Symbol(), volume, price, tp, profit)) {
        Print("❌ Cannot calculate profit");
        return false;
    }
    
    return true;
}

void UpdateTrailingStop()
{
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

void MoveToBreakEven()
{
    if(!positionInfo.SelectByTicket(currentTicket)) return;
    
    double openPrice = positionInfo.PriceOpen();
    double currentSL = positionInfo.StopLoss();
    
    // Check if SL is not already at break-even
    if(MathAbs(currentSL - openPrice) < Point() * 5) return;
    
    trade.PositionModify(currentTicket, openPrice, positionInfo.TakeProfit());
    Print("📊 Moved stop loss to break-even for ticket: ", currentTicket);
}

void TakePartialProfit()
{
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

void CloseCurrentPosition(string reason)
{
    if(trade.PositionClose(currentTicket)) {
        Print("🔒 Position closed: ", reason);
        positionOpen = false;
        currentTicket = 0;
    }
}

void HandlePositionClose(const MqlTradeTransaction& trans)
{
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

void MonitorRiskLevels()
{
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double drawdown = (balance - currentEquity) / balance * 100;
    
    if(drawdown > MAX_DRAWDOWN_PERCENT) {
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

void SendTradeNotification(string action, double price, double sl, double tp, double volume)
{
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

void UpdateStatistics()
{
    // Update daily statistics and other metrics
    botState.totalTrades = stats.executedTrades;
    botState.totalProfit = stats.totalPL;
    
    if(stats.executedTrades > 0) {
        botState.winningTrades = stats.profitableTrades;
    }
}

string GetDeInitReasonText(int reason)
{
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

void CheckSymbolProperties()
{
    Print("📊 Symbol Properties Check:");
    Print("   Symbol: ", Symbol());
    Print("   Digits: ", Digits());
    Print("   Point: ", Point());
    Print("   Spread: ", POINTS_TO_PIPS(SymbolInfoInteger(Symbol(), SYMBOL_SPREAD)), " pips");
    Print("   Trade Mode: ", SymbolInfoInteger(Symbol(), SYMBOL_TRADE_MODE));
    Print("   Execution Mode: ", SymbolInfoInteger(Symbol(), SYMBOL_TRADE_EXEMODE));
}

void CheckTradingLimits()
{
    double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    int stopLevel = (int)SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
    
    Print("📊 Trading Limits Check:");
    Print("   Min Lot: ", minLot);
    Print("   Max Lot: ", maxLot);
    Print("   Lot Step: ", stepLot);
    Print("   Stop Level: ", stopLevel, " points");
    
    if(InpLotSize < minLot || InpLotSize > maxLot) {
        Print("⚠️ WARNING: Input lot size ", InpLotSize, " is outside broker limits");
    }
}

void CheckAccountPermissions()
{
    Print("📊 Account Permissions Check:");
    Print("   Terminal Trading: ", TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? "Allowed" : "Disabled");
    Print("   EA Trading: ", MQLInfoInteger(MQL_TRADE_ALLOWED) ? "Allowed" : "Disabled");
    Print("   Account Trading: ", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) ? "Allowed" : "Disabled");
    Print("   Expert Trading: ", AccountInfoInteger(ACCOUNT_TRADE_EXPERT) ? "Allowed" : "Disabled");
    Print("   Account Type: ", AccountInfoInteger(ACCOUNT_TRADE_MODE));
    Print("   Balance: ", AccountInfoDouble(ACCOUNT_BALANCE));
    Print("   Equity: ", AccountInfoDouble(ACCOUNT_EQUITY));
    Print("   Free Margin: ", AccountInfoDouble(ACCOUNT_MARGIN_FREE));
}

void ResetBotState()
{
    ArrayInitialize(botState.fibLevels, 0);
    botState.truePosition = false;
    botState.hasTouched705Up = false;
    botState.hasTouched705Down = false;
    botState.hasValidFib = false;
    botState.fibCreationTime = 0;
    botState.fibStartPrice = 0;
    botState.fibEndPrice = 0;
    
    // Clear touched points
    ZeroMemory(botState.lastTouched705PointUp);
    ZeroMemory(botState.lastTouched705PointDown);
    
    Print("🔄 Bot state reset");
}

void CalculateFibonacci(double startPrice, double endPrice)
{
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

void HandleFibonacciTouch(int currentIdx, bool isBullish)
{
    string currentStatus = IS_BULLISH_CANDLE(rates, currentIdx) ? "bullish" : "bearish";
    
    if(isBullish) {
        if(!botState.hasTouched705Up) {
            Print("🎯 First touch of 0.705 level (bullish setup)");
            botState.lastTouched705PointUp = rates[currentIdx];
            botState.hasTouched705Up = true;
        }
        else {
            string lastStatus = IS_BULLISH_CANDLE(botState.lastTouched705PointUp, 0) ? "bullish" : "bearish";
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
            string lastStatus = IS_BULLISH_CANDLE(botState.lastTouched705PointDown, 0) ? "bullish" : "bearish";
            if(currentStatus != lastStatus) {
                Print("✅ Second touch of 0.705 level with different candle type - ENTRY SIGNAL!");
                botState.truePosition = true;
            }
        }
    }
}

void ProcessFibonacciUpdates(int currentIdx)
{
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

void MonitorPositions()
{
    int totalPositions = PositionsTotal();
    
    if(totalPositions == 0 && positionOpen) {
        Print("🏁 Position closed by market");
        positionOpen = false;
        currentTicket = 0;
    }
    
    // Additional position monitoring logic can be added here
}

#endif // SWING_TRADING_HELPERS_MQH
//+------------------------------------------------------------------+
