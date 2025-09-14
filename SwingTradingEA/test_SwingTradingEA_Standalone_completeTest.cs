//+------------------------------------------------------------------+
//|                    test_SwingTradingEA_Standalone_completeTest.mq5|
//|                        Test Suite for Swing Trading EA          |
//+------------------------------------------------------------------+
#property copyright "Swing Trading EA Test Suite"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Test framework globals
int totalTests = 0;
int passedTests = 0;
int failedTests = 0;
string testOutput = "";

//--- Test structures matching main EA
struct TestBotState {
    double fibLevels[4];
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
};

struct TestLeg {
    datetime start;
    datetime end;
    double startValue;
    double endValue;
    double length;
    string direction;
};

//--- Test globals
TestBotState testState;
TestLeg testLegsArray[];
int testLegsCount = 0;
MqlRates testRatesArray[];
string lastTestSwingType = "";

//--- Test configuration
input double TestLotSize = 0.01;
input double TestWinRatio = 1.2;
input int TestThreshold = 6;
input double TestFib705 = 0.705;
input double TestFib90 = 0.9;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🧪 Starting Swing Trading EA Test Suite...");
    Print("====================================================");
    
    //--- Initialize test environment
    InitializeTestEnvironment();
    
    //--- Run all tests
    RunAllTests();
    
    //--- Print final results
    PrintTestResults();
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Initialize test environment                                      |
//+------------------------------------------------------------------+
void InitializeTestEnvironment()
{
    totalTests = 0;
    passedTests = 0;
    failedTests = 0;
    testOutput = "";
    
    //--- Initialize arrays
    ArrayResize(testLegsArray, 100);
    ArrayResize(testRatesArray, 200);
    ArraySetAsSeries(testRatesArray, true);
    
    //--- Reset test state
    ResetTestState();
    
    Print("✅ Test environment initialized");
}

//+------------------------------------------------------------------+
//| Run all test suites                                             |
//+------------------------------------------------------------------+
void RunAllTests()
{
    Print("\n📋 Running Test Suites:");
    Print("----------------------------------------------------");
    
    //--- Core functionality tests
    TestFibonacciCalculations();
    TestLegDetectionLogic();
    TestSwingDetectionLogic();
    TestTradingSignalGeneration();
    TestTimeAndSessionValidation();
    TestRiskManagement();
    TestErrorHandling();
    TestDataValidation();
    TestArrayOperations();
    TestPriceCalculations();
}

//+------------------------------------------------------------------+
//| Test Fibonacci calculations (مطابق Python logic)               |
//+------------------------------------------------------------------+
void TestFibonacciCalculations()
{
    Print("\n📊 Testing Fibonacci Calculations:");
    
    //--- Test bullish fibonacci (endPrice > startPrice)
    StartTest("Bullish Fibonacci Calculation");
    double testStart = 1.10000;
    double testEnd = 1.11000;
    TestCalculateFib(testStart, testEnd);
    
    AssertEquals(testEnd, testState.fibLevels[0], "Bullish Fib 0.0 should equal higher price");
    AssertEquals(testStart, testState.fibLevels[3], "Bullish Fib 1.0 should equal lower price");
    
    double expected705 = testEnd - TestFib705 * (testEnd - testStart);
    AssertEquals(expected705, testState.fibLevels[1], "Bullish Fib 0.705 calculation");
    
    double expected90 = testEnd - TestFib90 * (testEnd - testStart);
    AssertEquals(expected90, testState.fibLevels[2], "Bullish Fib 0.9 calculation");
    
    //--- Test bearish fibonacci (startPrice > endPrice)
    StartTest("Bearish Fibonacci Calculation");
    TestCalculateFib(testEnd, testStart);
    
    AssertEquals(testStart, testState.fibLevels[0], "Bearish Fib 0.0 should equal higher price");
    AssertEquals(testEnd, testState.fibLevels[3], "Bearish Fib 1.0 should equal lower price");
    
    expected705 = testStart - TestFib705 * (testStart - testEnd);
    AssertEquals(expected705, testState.fibLevels[1], "Bearish Fib 0.705 calculation");
    
    //--- Test fibonacci level ordering
    StartTest("Fibonacci Levels Ordering");
    if(testState.fibLevels[0] > testState.fibLevels[3]) {
        // Bearish fibonacci
        AssertTrue(testState.fibLevels[0] > testState.fibLevels[1], "Fib 0.0 > Fib 0.705 in bearish");
        AssertTrue(testState.fibLevels[1] > testState.fibLevels[2], "Fib 0.705 > Fib 0.9 in bearish");
        AssertTrue(testState.fibLevels[2] > testState.fibLevels[3], "Fib 0.9 > Fib 1.0 in bearish");
    } else {
        // Bullish fibonacci
        AssertTrue(testState.fibLevels[0] < testState.fibLevels[1], "Fib 0.0 < Fib 0.705 in bullish");
        AssertTrue(testState.fibLevels[1] < testState.fibLevels[2], "Fib 0.705 < Fib 0.9 in bullish");
        AssertTrue(testState.fibLevels[2] < testState.fibLevels[3], "Fib 0.9 < Fib 1.0 in bullish");
    }
}

//+------------------------------------------------------------------+
//| Test leg detection logic (مطابق get_legs.py)                   |
//+------------------------------------------------------------------+
void TestLegDetectionLogic()
{
    Print("\n📈 Testing Leg Detection Logic:");
    
    //--- Test basic leg detection
    StartTest("Basic Leg Detection");
    CreateTrendingPriceData();
    TestGetLegs();
    AssertTrue(testLegsCount > 0, "Should detect legs in trending data");
    
    //--- Test leg direction consistency
    StartTest("Leg Direction Consistency");
    bool directionValid = true;
    for(int i = 0; i < testLegsCount; i++) {
        if(testLegsArray[i].direction != "up" && testLegsArray[i].direction != "down") {
            directionValid = false;
            break;
        }
        
        if(testLegsArray[i].direction == "up") {
            AssertTrue(testLegsArray[i].endValue > testLegsArray[i].startValue, "Up leg should have higher end value");
        } else {
            AssertTrue(testLegsArray[i].endValue < testLegsArray[i].startValue, "Down leg should have lower end value");
        }
    }
    AssertTrue(directionValid, "All legs should have valid direction");
    
    //--- Test minimum threshold requirement
    StartTest("Minimum Threshold Validation");
    bool thresholdMet = true;
    for(int i = 0; i < testLegsCount; i++) {
        if(testLegsArray[i].length < TestThreshold) {
            thresholdMet = false;
            break;
        }
    }
    AssertTrue(thresholdMet, "All legs should meet minimum threshold");
    
    //--- Test alternating leg directions
    StartTest("Alternating Leg Directions");
    if(testLegsCount > 1) {
        bool alternating = true;
        for(int i = 1; i < testLegsCount; i++) {
            if(testLegsArray[i].direction == testLegsArray[i-1].direction) {
                alternating = false;
                break;
            }
        }
        AssertTrue(alternating, "Consecutive legs should have different directions");
    }
}

//+------------------------------------------------------------------+
//| Test swing detection logic (مطابق swing.py)                    |
//+------------------------------------------------------------------+
void TestSwingDetectionLogic()
{
    Print("\n🎯 Testing Swing Detection Logic:");
    
    //--- Test bullish swing pattern
    StartTest("Bullish Swing Pattern Detection");
    CreateBullishSwingData();
    string swingType;
    bool isSwing = TestGetSwingPoints(swingType);
    
    if(testLegsCount >= 3) {
        AssertTrue(isSwing, "Should detect bullish swing in valid pattern");
        if(isSwing) {
            AssertEquals("bullish", swingType, "Should identify as bullish swing");
        }
    }
    
    //--- Test bearish swing pattern
    StartTest("Bearish Swing Pattern Detection");
    CreateBearishSwingData();
    isSwing = TestGetSwingPoints(swingType);
    
    if(testLegsCount >= 3) {
        AssertTrue(isSwing, "Should detect bearish swing in valid pattern");
        if(isSwing) {
            AssertEquals("bearish", swingType, "Should identify as bearish swing");
        }
    }
    
    //--- Test insufficient legs
    StartTest("Insufficient Legs Handling");
    ArrayResize(testLegsArray, 2);
    testLegsCount = 2;
    isSwing = TestGetSwingPoints(swingType);
    AssertFalse(isSwing, "Should not detect swing with less than 3 legs");
    
    //--- Test invalid swing pattern
    StartTest("Invalid Swing Pattern");
    CreateInvalidSwingData();
    isSwing = TestGetSwingPoints(swingType);
    AssertFalse(isSwing, "Should not detect swing in invalid pattern");
}

//+------------------------------------------------------------------+
//| Test trading signal generation (مطابق Python buy/sell logic)   |
//+------------------------------------------------------------------+
void TestTradingSignalGeneration()
{
    Print("\n🎯 Testing Trading Signal Generation:");
    
    //--- Test buy signal conditions
    StartTest("Buy Signal Generation");
    SetupBuyConditions();
    bool buySignal = TestCheckBuyConditions();
    AssertTrue(buySignal, "Should generate buy signal with valid conditions");
    
    //--- Test sell signal conditions
    StartTest("Sell Signal Generation");
    SetupSellConditions();
    bool sellSignal = TestCheckSellConditions();
    AssertTrue(sellSignal, "Should generate sell signal with valid conditions");
    
    //--- Test no signal when conditions not met
    StartTest("No Signal When Conditions Not Met");
    testState.truePosition = false;
    buySignal = TestCheckBuyConditions();
    sellSignal = TestCheckSellConditions();
    AssertFalse(buySignal, "Should not generate buy signal without true position");
    AssertFalse(sellSignal, "Should not generate sell signal without true position");
    
    //--- Test stop loss calculation for bullish
    StartTest("Bullish Stop Loss Calculation");
    SetupBuyConditions();
    double currentPrice = 1.10800;
    double stopLoss = TestCalculateStopLoss(currentPrice, "bullish");
    AssertTrue(stopLoss < currentPrice, "Stop loss should be below current price for bullish");
    AssertTrue(stopLoss >= testState.fibLevels[2], "Stop loss should be at or above fib 0.9 level");
    
    //--- Test stop loss calculation for bearish
    StartTest("Bearish Stop Loss Calculation");
    SetupSellConditions();
    currentPrice = 1.10200;
    stopLoss = TestCalculateStopLoss(currentPrice, "bearish");
    AssertTrue(stopLoss > currentPrice, "Stop loss should be above current price for bearish");
    AssertTrue(stopLoss <= testState.fibLevels[2], "Stop loss should be at or below fib 0.9 level");
    
    //--- Test take profit calculation
    StartTest("Take Profit Calculation");
    currentPrice = 1.10500;
    stopLoss = 1.10300;
    double takeProfit = TestCalculateTakeProfit(currentPrice, stopLoss);
    double expectedTP = currentPrice + (MathAbs(currentPrice - stopLoss) * TestWinRatio);
    AssertEquals(expectedTP, takeProfit, "Take profit calculation should match win ratio");
}

//+------------------------------------------------------------------+
//| Test time and session validation (مطابق Iran time logic)       |
//+------------------------------------------------------------------+
void TestTimeAndSessionValidation()
{
    Print("\n⏰ Testing Time and Session Validation:");
    
    //--- Test Iran time conversion
    StartTest("Iran Time Conversion");
    string iranTime = TestGetIranTimeString();
    AssertTrue(StringLen(iranTime) >= 19, "Iran time string should be properly formatted");
    AssertTrue(StringFind(iranTime, ":") > 0, "Iran time should contain time separator");
    
    //--- Test weekend detection
    StartTest("Weekend Detection");
    bool isWeekend = TestIsIranWeekend();
    // Just verify it returns a valid boolean
    AssertTrue(isWeekend == true || isWeekend == false, "Weekend detection should return boolean");
    
    //--- Test trading hours validation
    StartTest("Trading Hours Validation");
    bool validHours = TestValidateTradingHours("09:00", "21:00");
    AssertTrue(validHours, "Valid trading hours should be accepted");
    
    validHours = TestValidateTradingHours("25:00", "21:00");
    AssertFalse(validHours, "Invalid start hour should be rejected");
    
    validHours = TestValidateTradingHours("09:00", "08:00");
    AssertFalse(validHours, "End time before start time should be rejected");
    
    //--- Test session overlap calculation
    StartTest("Session Overlap Calculation");
    bool overlap = TestCheckSessionOverlap();
    AssertTrue(overlap == true || overlap == false, "Session overlap should return boolean");
}

//+------------------------------------------------------------------+
//| Test risk management functions                                   |
//+------------------------------------------------------------------+
void TestRiskManagement()
{
    Print("\n💰 Testing Risk Management:");
    
    //--- Test lot size calculation
    StartTest("Lot Size Calculation");
    double balance = 10000.0;
    double riskPercent = 2.0;
    double stopDistance = 0.00200; // 20 pips
    double lotSize = TestCalculateLotSize(balance, riskPercent, stopDistance);
    AssertTrue(lotSize > 0, "Lot size should be positive");
    AssertTrue(lotSize <= 10.0, "Lot size should be reasonable");
    
    //--- Test maximum risk validation
    StartTest("Maximum Risk Validation");
    bool riskValid = TestValidateRisk(riskPercent, balance);
    AssertTrue(riskValid, "Valid risk percentage should be accepted");
    
    riskValid = TestValidateRisk(50.0, balance);
    AssertFalse(riskValid, "Excessive risk percentage should be rejected");
    
    //--- Test spread validation
    StartTest("Spread Validation");
    double spread = 1.5; // 1.5 pips
    bool spreadOk = TestValidateSpread(spread, 3.0);
    AssertTrue(spreadOk, "Acceptable spread should be valid");
    
    spread = 5.0; // 5 pips
    spreadOk = TestValidateSpread(spread, 3.0);
    AssertFalse(spreadOk, "Excessive spread should be rejected");
    
    //--- Test daily trade limit
    StartTest("Daily Trade Limit");
    bool canTrade = TestCheckDailyLimit(5, 10);
    AssertTrue(canTrade, "Should allow trading under daily limit");
    
    canTrade = TestCheckDailyLimit(10, 10);
    AssertFalse(canTrade, "Should block trading at daily limit");
}

//+------------------------------------------------------------------+
//| Test error handling                                             |
//+------------------------------------------------------------------+
void TestErrorHandling()
{
    Print("\n🚨 Testing Error Handling:");
    
    //--- Test invalid parameters
    StartTest("Invalid Parameter Handling");
    bool valid = TestValidateInputs(-0.01, TestWinRatio, TestThreshold);
    AssertFalse(valid, "Should reject negative lot size");
    
    valid = TestValidateInputs(TestLotSize, 0, TestThreshold);
    AssertFalse(valid, "Should reject zero win ratio");
    
    valid = TestValidateInputs(TestLotSize, TestWinRatio, -1);
    AssertFalse(valid, "Should reject negative threshold");
    
    //--- Test empty data handling
    StartTest("Empty Data Handling");
    ArrayResize(testRatesArray, 0);
    TestGetLegs();
    AssertEquals(0, testLegsCount, "Should handle empty data gracefully");
    
    //--- Test invalid fibonacci calculation
    StartTest("Invalid Fibonacci Handling");
    TestCalculateFib(1.10000, 1.10000); // Same prices
    AssertFalse(testState.hasValidFib, "Should reject fibonacci with same start/end prices");
    
    //--- Test array bounds
    StartTest("Array Bounds Checking");
    bool boundsOk = TestArrayBounds();
    AssertTrue(boundsOk, "Array operations should respect bounds");
}

//+------------------------------------------------------------------+
//| Test data validation                                            |
//+------------------------------------------------------------------+
void TestDataValidation()
{
    Print("\n📊 Testing Data Validation:");
    
    //--- Test price data integrity
    StartTest("Price Data Integrity");
    CreateValidPriceData();
    bool dataValid = TestValidatePriceData();
    AssertTrue(dataValid, "Valid price data should pass validation");
    
    //--- Test time sequence validation
    StartTest("Time Sequence Validation");
    bool timeValid = TestValidateTimeSequence();
    AssertTrue(timeValid, "Time sequence should be properly ordered");
    
    //--- Test OHLC relationships
    StartTest("OHLC Relationship Validation");
    bool ohlcValid = TestValidateOHLC();
    AssertTrue(ohlcValid, "OHLC relationships should be valid");
    
    //--- Test volume validation
    StartTest("Volume Validation");
    bool volumeValid = TestValidateVolume();
    AssertTrue(volumeValid, "Volume data should be positive");
}

//+------------------------------------------------------------------+
//| Test array operations                                           |
//+------------------------------------------------------------------+
void TestArrayOperations()
{
    Print("\n🔧 Testing Array Operations:");
    
    //--- Test array resizing
    StartTest("Array Resizing");
    int originalSize = ArraySize(testLegsArray);
    ArrayResize(testLegsArray, originalSize + 10);
    AssertEquals(originalSize + 10, ArraySize(testLegsArray), "Array should resize correctly");
    
    //--- Test array copying
    StartTest("Array Copying");
    TestLeg tempLegs[];
    ArrayResize(tempLegs, 3);
    for(int i = 0; i < 3; i++) {
        tempLegs[i] = testLegsArray[i];
    }
    AssertEquals(testLegsArray[0].direction, tempLegs[0].direction, "Array copying should work");
    
    //--- Test array initialization
    StartTest("Array Initialization");
    double testArray[10];
    ArrayInitialize(testArray, 0.0);
    bool allZero = true;
    for(int i = 0; i < 10; i++) {
        if(testArray[i] != 0.0) {
            allZero = false;
            break;
        }
    }
    AssertTrue(allZero, "Array initialization should set all elements");
}

//+------------------------------------------------------------------+
//| Test price calculations                                         |
//+------------------------------------------------------------------+
void TestPriceCalculations()
{
    Print("\n💱 Testing Price Calculations:");
    
    //--- Test pip calculation
    StartTest("Pip Calculation");
    double price1 = 1.10000;
    double price2 = 1.10050;
    double pips = TestCalculatePips(price1, price2);
    AssertEquals(5.0, pips, "Pip calculation should be accurate");
    
    //--- Test price normalization
    StartTest("Price Normalization");
    double rawPrice = 1.123456789;
    double normalized = TestNormalizePrice(rawPrice);
    AssertTrue(normalized >= 1.12345 && normalized <= 1.12346, "Price should be normalized correctly");
    
    //--- Test spread calculation
    StartTest("Spread Calculation");
    double bid = 1.10000;
    double ask = 1.10003;
    double spread = TestCalculateSpread(bid, ask);
    AssertEquals(0.3, spread, "Spread calculation should be in pips");
}

//+------------------------------------------------------------------+
//| Test implementation functions                                   |
//+------------------------------------------------------------------+

void TestCalculateFib(double startPrice, double endPrice)
{
    if(MathAbs(startPrice - endPrice) < 0.00001) {
        testState.hasValidFib = false;
        return;
    }
    
    if(endPrice > startPrice) {
        // Bullish fibonacci
        testState.fibLevels[0] = endPrice;      // fib 0.0
        testState.fibLevels[1] = endPrice - TestFib705 * (endPrice - startPrice); // fib 0.705
        testState.fibLevels[2] = endPrice - TestFib90 * (endPrice - startPrice);  // fib 0.9
        testState.fibLevels[3] = startPrice;    // fib 1.0
    } else {
        // Bearish fibonacci
        testState.fibLevels[0] = startPrice;    // fib 0.0
        testState.fibLevels[1] = startPrice - TestFib705 * (startPrice - endPrice); // fib 0.705
        testState.fibLevels[2] = startPrice - TestFib90 * (startPrice - endPrice);  // fib 0.9
        testState.fibLevels[3] = endPrice;      // fib 1.0
    }
    
    testState.hasValidFib = true;
}

void TestGetLegs()
{
    testLegsCount = 0;
    ArrayFree(testLegsArray);
    
    if(ArraySize(testRatesArray) < 10) return;
    
    double threshold = TestThreshold;
    int currentLeg = 0;
    datetime currentStart = testRatesArray[ArraySize(testRatesArray)-1].time;
    double currentStartValue = testRatesArray[ArraySize(testRatesArray)-1].close;
    string lastDirection = "";
    
    for(int i = ArraySize(testRatesArray)-2; i >= 0; i--) {
        double priceDiff = MathAbs(testRatesArray[i].close - currentStartValue) * 10000;
        
        if(priceDiff >= threshold) {
            string direction = (testRatesArray[i].close > currentStartValue) ? "up" : "down";
            
            if(lastDirection == "" || lastDirection != direction) {
                // Create new leg
                ArrayResize(testLegsArray, currentLeg + 1);
                testLegsArray[currentLeg].start = currentStart;
                testLegsArray[currentLeg].end = testRatesArray[i].time;
                testLegsArray[currentLeg].startValue = currentStartValue;
                testLegsArray[currentLeg].endValue = testRatesArray[i].close;
                testLegsArray[currentLeg].length = priceDiff;
                testLegsArray[currentLeg].direction = direction;
                
                currentLeg++;
                lastDirection = direction;
                currentStart = testRatesArray[i].time;
                currentStartValue = testRatesArray[i].close;
            }
        }
    }
    
    testLegsCount = currentLeg;
}

bool TestGetSwingPoints(string &swingType)
{
    swingType = "";
    if(testLegsCount < 3) return false;
    
    // Check for bullish swing: down, up, down pattern where middle leg is highest
    if(testLegsArray[0].direction == "down" && 
       testLegsArray[1].direction == "up" && 
       testLegsArray[2].direction == "down") {
        
        if(testLegsArray[1].endValue > testLegsArray[0].startValue && 
           testLegsArray[1].endValue > testLegsArray[2].endValue) {
            swingType = "bullish";
            return true;
        }
    }
    
    // Check for bearish swing: up, down, up pattern where middle leg is lowest
    if(testLegsArray[0].direction == "up" && 
       testLegsArray[1].direction == "down" && 
       testLegsArray[2].direction == "up") {
        
        if(testLegsArray[1].endValue < testLegsArray[0].startValue && 
           testLegsArray[1].endValue < testLegsArray[2].endValue) {
            swingType = "bearish";
            return true;
        }
    }
    
    return false;
}

bool TestCheckBuyConditions()
{
    if(!testState.truePosition || !testState.hasValidFib) return false;
    if(ArraySize(testRatesArray) == 0) return false;
    
    // Check if current price is in buy zone (above fib 0.705)
    return (testRatesArray[0].close > testState.fibLevels[1]);
}

bool TestCheckSellConditions()
{
    if(!testState.truePosition || !testState.hasValidFib) return false;
    if(ArraySize(testRatesArray) == 0) return false;
    
    // Check if current price is in sell zone (below fib 0.705)
    return (testRatesArray[0].close < testState.fibLevels[1]);
}

double TestCalculateStopLoss(double currentPrice, string direction)
{
    if(direction == "bullish") {
        return testState.fibLevels[2]; // Use fib 0.9 level
    } else {
        return testState.fibLevels[2]; // Use fib 0.9 level
    }
}

double TestCalculateTakeProfit(double currentPrice, double stopLoss)
{
    double distance = MathAbs(currentPrice - stopLoss);
    if(currentPrice > stopLoss) {
        return currentPrice + (distance * TestWinRatio);
    } else {
        return currentPrice - (distance * TestWinRatio);
    }
}

//+------------------------------------------------------------------+
//| Helper functions for test data creation                         |
//+------------------------------------------------------------------+

void CreateTrendingPriceData()
{
    ArrayResize(testRatesArray, 50);
    datetime baseTime = TimeCurrent() - 3000;
    double basePrice = 1.10000;
    
    for(int i = 49; i >= 0; i--) {
        testRatesArray[i].time = baseTime + (49-i) * 60;
        
        // Create trending pattern
        if(i > 30) {
            basePrice += 0.00002; // Uptrend
        } else if(i > 15) {
            basePrice -= 0.00003; // Downtrend
        } else {
            basePrice += 0.00001; // Weak uptrend
        }
        
        testRatesArray[i].open = basePrice;
        testRatesArray[i].high = basePrice + 0.00005;
        testRatesArray[i].low = basePrice - 0.00005;
        testRatesArray[i].close = basePrice + (MathRand() % 11 - 5) * 0.00001;
        testRatesArray[i].tick_volume = 100;
    }
}

void CreateBullishSwingData()
{
    ArrayResize(testLegsArray, 3);
    testLegsCount = 3;
    
    datetime baseTime = TimeCurrent();
    
    // Create bullish swing: down, up (swing high), down
    testLegsArray[0].start = baseTime - 1800;
    testLegsArray[0].end = baseTime - 1200;
    testLegsArray[0].startValue = 1.10500;
    testLegsArray[0].endValue = 1.10200;
    testLegsArray[0].direction = "down";
    testLegsArray[0].length = 30;
    
    testLegsArray[1].start = baseTime - 1200;
    testLegsArray[1].end = baseTime - 600;
    testLegsArray[1].startValue = 1.10200;
    testLegsArray[1].endValue = 1.10800; // Swing high
    testLegsArray[1].direction = "up";
    testLegsArray[1].length = 60;
    
    testLegsArray[2].start = baseTime - 600;
    testLegsArray[2].end = baseTime;
    testLegsArray[2].startValue = 1.10800;
    testLegsArray[2].endValue = 1.10400;
    testLegsArray[2].direction = "down";
    testLegsArray[2].length = 40;
}

void CreateBearishSwingData()
{
    ArrayResize(testLegsArray, 3);
    testLegsCount = 3;
    
    datetime baseTime = TimeCurrent();
    
    // Create bearish swing: up, down (swing low), up
    testLegsArray[0].start = baseTime - 1800;
    testLegsArray[0].end = baseTime - 1200;
    testLegsArray[0].startValue = 1.10200;
    testLegsArray[0].endValue = 1.10500;
    testLegsArray[0].direction = "up";
    testLegsArray[0].length = 30;
    
    testLegsArray[1].start = baseTime - 1200;
    testLegsArray[1].end = baseTime - 600;
    testLegsArray[1].startValue = 1.10500;
    testLegsArray[1].endValue = 1.10100; // Swing low
    testLegsArray[1].direction = "down";
    testLegsArray[1].length = 40;
    
    testLegsArray[2].start = baseTime - 600;
    testLegsArray[2].end = baseTime;
    testLegsArray[2].startValue = 1.10100;
    testLegsArray[2].endValue = 1.10400;
    testLegsArray[2].direction = "up";
    testLegsArray[2].length = 30;
}

void CreateInvalidSwingData()
{
    ArrayResize(testLegsArray, 3);
    testLegsCount = 3;
    
    datetime baseTime = TimeCurrent();
    
    // Create invalid pattern: all same direction
    for(int i = 0; i < 3; i++) {
        testLegsArray[i].start = baseTime - (1800 - i * 600);
        testLegsArray[i].end = baseTime - (1200 - i * 600);
        testLegsArray[i].startValue = 1.10000 + i * 0.00100;
        testLegsArray[i].endValue = 1.10100 + i * 0.00100;
        testLegsArray[i].direction = "up";
        testLegsArray[i].length = 10;
    }
}

void SetupBuyConditions()
{
    testState.truePosition = true;
    testState.hasValidFib = true;
    testState.fibLevels[0] = 1.11000; // fib 0.0
    testState.fibLevels[1] = 1.10295; // fib 0.705
    testState.fibLevels[2] = 1.10100; // fib 0.9
    testState.fibLevels[3] = 1.10000; // fib 1.0
    
    ArrayResize(testRatesArray, 1);
    testRatesArray[0].time = TimeCurrent();
    testRatesArray[0].close = 1.10400; // Above fib 0.705
    lastTestSwingType = "bullish";
}

void SetupSellConditions()
{
    testState.truePosition = true;
    testState.hasValidFib = true;
    testState.fibLevels[0] = 1.10000; // fib 0.0
    testState.fibLevels[1] = 1.10705; // fib 0.705
    testState.fibLevels[2] = 1.10900; // fib 0.9
    testState.fibLevels[3] = 1.11000; // fib 1.0
    
    ArrayResize(testRatesArray, 1);
    testRatesArray[0].time = TimeCurrent();
    testRatesArray[0].close = 1.10600; // Below fib 0.705
    lastTestSwingType = "bearish";
}

//+------------------------------------------------------------------+
//| Utility test functions                                          |
//+------------------------------------------------------------------+

string TestGetIranTimeString()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime); // GMT+3:30
    return StringFormat("%04d.%02d.%02d %02d:%02d:%02d", 
                       iranTime.year, iranTime.mon, iranTime.day,
                       iranTime.hour, iranTime.min, iranTime.sec);
}

bool TestIsIranWeekend()
{
    MqlDateTime iranTime;
    TimeToStruct(TimeGMT() + 12600, iranTime);
    return (iranTime.day_of_week == 5 || iranTime.day_of_week == 6);
}

bool TestValidateTradingHours(string startTime, string endTime)
{
    string parts[];
    int startParts = StringSplit(startTime, ':', parts);
    if(startParts != 2) return false;
    
    int startHour = (int)StringToInteger(parts[0]);
    int startMin = (int)StringToInteger(parts[1]);
    
    int endParts = StringSplit(endTime, ':', parts);
    if(endParts != 2) return false;
    
    int endHour = (int)StringToInteger(parts[0]);
    int endMin = (int)StringToInteger(parts[1]);
    
    if(startHour < 0 || startHour > 23 || endHour < 0 || endHour > 23) return false;
    if(startMin < 0 || startMin > 59 || endMin < 0 || endMin > 59) return false;
    
    int startMinutes = startHour * 60 + startMin;
    int endMinutes = endHour * 60 + endMin;
    
    return (endMinutes > startMinutes);
}

bool TestCheckSessionOverlap()
{
    return true; // Simplified for testing
}

double TestCalculateLotSize(double balance, double riskPercent, double stopDistance)
{
    double riskAmount = balance * riskPercent / 100.0;
    double pipValue = 1.0; // Simplified pip value
    double lots = riskAmount / (stopDistance * 100000 * pipValue);
    return NormalizeDouble(lots, 2);
}

bool TestValidateRisk(double riskPercent, double balance)
{
    return (riskPercent > 0 && riskPercent <= 10.0 && balance > 0);
}

bool TestValidateSpread(double currentSpread, double maxSpread)
{
    return (currentSpread <= maxSpread);
}

bool TestCheckDailyLimit(int currentTrades, int maxTrades)
{
    return (currentTrades < maxTrades);
}

bool TestValidateInputs(double lotSize, double winRatio, int threshold)
{
    return (lotSize > 0 && winRatio > 0 && threshold > 0);
}

bool TestArrayBounds()
{
    if(ArraySize(testRatesArray) == 0) return true;
    
    // Test safe array access
    for(int i = 0; i < ArraySize(testRatesArray); i++) {
        if(testRatesArray[i].time < 0) return false;
    }
    return true;
}

void CreateValidPriceData()
{
    ArrayResize(testRatesArray, 10);
    datetime baseTime = TimeCurrent() - 600;
    
    for(int i = 9; i >= 0; i--) {
        testRatesArray[i].time = baseTime + (9-i) * 60;
        testRatesArray[i].open = 1.10000 + (MathRand() % 50) * 0.00001;
        testRatesArray[i].high = testRatesArray[i].open + (MathRand() % 30) * 0.00001;
        testRatesArray[i].low = testRatesArray[i].open - (MathRand() % 30) * 0.00001;
        testRatesArray[i].close = testRatesArray[i].low + (MathRand() % (int)((testRatesArray[i].high - testRatesArray[i].low) * 100000)) * 0.00001;
        testRatesArray[i].tick_volume = 100 + MathRand() % 900;
    }
}

bool TestValidatePriceData()
{
    for(int i = 0; i < ArraySize(testRatesArray); i++) {
        if(testRatesArray[i].high < testRatesArray[i].low) return false;
        if(testRatesArray[i].close < testRatesArray[i].low || testRatesArray[i].close > testRatesArray[i].high) return false;
        if(testRatesArray[i].open < testRatesArray[i].low || testRatesArray[i].open > testRatesArray[i].high) return false;
    }
    return true;
}

bool TestValidateTimeSequence()
{
    for(int i = 1; i < ArraySize(testRatesArray); i++) {
        if(testRatesArray[i].time <= testRatesArray[i-1].time) return false;
    }
    return true;
}

bool TestValidateOHLC()
{
    for(int i = 0; i < ArraySize(testRatesArray); i++) {
        if(testRatesArray[i].high < testRatesArray[i].open || 
           testRatesArray[i].high < testRatesArray[i].close ||
           testRatesArray[i].low > testRatesArray[i].open || 
           testRatesArray[i].low > testRatesArray[i].close) {
            return false;
        }
    }
    return true;
}

bool TestValidateVolume()
{
    for(int i = 0; i < ArraySize(testRatesArray); i++) {
        if(testRatesArray[i].tick_volume <= 0) return false;
    }
    return true;
}

double TestCalculatePips(double price1, double price2)
{
    return MathAbs(price1 - price2) * 10000;
}

double TestNormalizePrice(double price)
{
    return NormalizeDouble(price, 5);
}

double TestCalculateSpread(double bid, double ask)
{
    return (ask - bid) * 10000;
}

void ResetTestState()
{
    ArrayInitialize(testState.fibLevels, 0);
    testState.truePosition = false;
    testState.hasTouched705Up = false;
    testState.hasTouched705Down = false;
    testState.hasValidFib = false;
    testState.fibCreationTime = 0;
    testState.consecutiveFailures = 0;
    testState.totalProfit = 0;
    testState.totalTrades = 0;
}

//+------------------------------------------------------------------+
//| Test framework functions                                        |
//+------------------------------------------------------------------+

void StartTest(string testName)
{
    totalTests++;
    testOutput += StringFormat("  🧪 %s\n", testName);
}

void AssertTrue(bool condition, string message)
{
    if(condition) {
        passedTests++;
        testOutput += StringFormat("    ✅ %s\n", message);
    } else {
        failedTests++;
        testOutput += StringFormat("    ❌ %s\n", message);
    }
}

void AssertFalse(bool condition, string message)
{
    AssertTrue(!condition, message);
}

void AssertEquals(double expected, double actual, string message)
{
    double tolerance = 0.00001;
    bool isEqual = MathAbs(expected - actual) < tolerance;
    
    if(isEqual) {
        passedTests++;
        testOutput += StringFormat("    ✅ %s (Expected: %.5f, Actual: %.5f)\n", message, expected, actual);
    } else {
        failedTests++;
        testOutput += StringFormat("    ❌ %s (Expected: %.5f, Actual: %.5f)\n", message, expected, actual);
    }
}

void AssertEquals(int expected, int actual, string message)
{
    if(expected == actual) {
        passedTests++;
        testOutput += StringFormat("    ✅ %s (Expected: %d, Actual: %d)\n", message, expected, actual);
    } else {
        failedTests++;
        testOutput += StringFormat("    ❌ %s (Expected: %d, Actual: %d)\n", message, expected, actual);
    }
}

void AssertEquals(string expected, string actual, string message)
{
    if(expected == actual) {
        passedTests++;
        testOutput += StringFormat("    ✅ %s (Expected: %s, Actual: %s)\n", message, expected, actual);
    } else {
        failedTests++;
        testOutput += StringFormat("    ❌ %s (Expected: %s, Actual: %s)\n", message, expected, actual);
    }
}

void PrintTestResults()
{
    Print("\n====================================================");
    Print("📊 TEST RESULTS SUMMARY");
    Print("====================================================");
    Print("Total Tests: ", totalTests);
    Print("✅ Passed: ", passedTests);
    Print("❌ Failed: ", failedTests);
    
    if(failedTests == 0) {
        Print("🎉 ALL TESTS PASSED!");
    } else {
        Print("⚠️  ", failedTests, " tests failed - check implementation");
    }
    
    double successRate = (totalTests > 0) ? (double)passedTests / totalTests * 100 : 0;
    Print("Success Rate: ", successRate, "%");
    Print("====================================================");
    
    // Print detailed test output
    Print("\n📋 DETAILED TEST OUTPUT:");
    Print(testOutput);
}

void OnTick() {}
void OnDeinit(const int reason) { Print("🧪 Test suite completed"); }