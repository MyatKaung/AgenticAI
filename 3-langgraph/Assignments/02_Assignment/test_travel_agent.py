#!/usr/bin/env python3
"""
Test script for AI Travel Agent & Expense Planner
This script validates the system functionality and demonstrates basic usage.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if environment is properly configured"""
    print("🔧 Testing Environment Configuration...")
    print("=" * 50)
    
    required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        value = os.getenv(key)
        if value:
            print(f"✅ {key}: {'*' * (len(value) - 4) + value[-4:]}")
        else:
            print(f"❌ {key}: Not found")
            missing_keys.append(key)
    
    optional_keys = ["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
    for key in optional_keys:
        value = os.getenv(key)
        if value:
            print(f"🔵 {key}: {'*' * (len(value) - 4) + value[-4:]} (Optional)")
        else:
            print(f"⚪ {key}: Not set (Optional)")
    
    if missing_keys:
        print(f"\n❌ Missing required API keys: {', '.join(missing_keys)}")
        print("Please check your .env file and add the missing keys.")
        return False
    else:
        print("\n✅ Environment configuration is complete!")
        return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n📦 Testing Module Imports...")
    print("=" * 50)
    
    try:
        from ai_travel_agent import AITravelAgent, TripRequest
        print("✅ AI Travel Agent modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def test_basic_functionality():
    """Test basic system functionality with a simple request"""
    print("\n🧪 Testing Basic Functionality...")
    print("=" * 50)
    
    try:
        from ai_travel_agent import AITravelAgent, TripRequest
        
        # Initialize agent
        print("🤖 Initializing AI Travel Agent...")
        agent = AITravelAgent()
        print("✅ Agent initialized successfully")
        
        # Create a simple test request
        print("\n📝 Creating test trip request...")
        test_trip = TripRequest(
            destination="London, UK",
            start_date="2024-08-01",
            end_date="2024-08-05",
            budget=2000.0,
            currency="USD",
            travelers=1,
            preferences=["museums", "history"]
        )
        print(f"✅ Test request created for {test_trip.destination}")
        
        # Test individual components
        print("\n🔍 Testing individual components...")
        
        # Test state initialization
        initial_state = {
            "trip_request": {
                "destination": test_trip.destination,
                "start_date": test_trip.start_date,
                "end_date": test_trip.end_date,
                "budget": test_trip.budget,
                "currency": test_trip.currency,
                "travelers": test_trip.travelers,
                "preferences": test_trip.preferences
            },
            "weather_info": {},
            "attractions": [],
            "hotels": [],
            "expenses": {},
            "currency_rates": {},
            "itinerary": [],
            "summary": "",
            "current_step": "start",
            "errors": []
        }
        print("✅ State initialization successful")
        
        # Test supervisor routing
        next_step = agent.supervisor.route_next(initial_state)
        print(f"✅ Supervisor routing works: {initial_state['current_step']} -> {next_step}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_quick_trip_planning():
    """Test quick trip planning with minimal API calls"""
    print("\n🚀 Testing Quick Trip Planning...")
    print("=" * 50)
    
    try:
        from ai_travel_agent import AITravelAgent, TripRequest
        
        # Create a simple trip request
        quick_trip = TripRequest(
            destination="New York, USA",
            start_date="2024-06-15",
            end_date="2024-06-18",
            budget=1500.0,
            currency="USD",
            travelers=1,
            preferences=["art", "food"]
        )
        
        print(f"🗽 Planning quick trip to {quick_trip.destination}...")
        print("⏱️ This may take 1-2 minutes for real-time data...")
        
        # Initialize agent and plan trip
        agent = AITravelAgent()
        result = agent.plan_trip(quick_trip)
        
        if "error" in result:
            print(f"❌ Trip planning failed: {result['error']}")
            return False
        else:
            print("✅ Trip planning completed successfully!")
            
            # Validate result structure
            required_keys = ["trip_request", "expenses", "summary"]
            for key in required_keys:
                if key in result:
                    print(f"✅ {key}: Present")
                else:
                    print(f"❌ {key}: Missing")
            
            # Display basic summary
            expenses = result.get("expenses", {})
            total_cost = expenses.get("total", 0)
            daily_budget = expenses.get("daily_budget", 0)
            
            print(f"\n💰 Total Cost: ${total_cost:.2f}")
            print(f"📅 Daily Budget: ${daily_budget:.2f}")
            
            # Save test results
            with open("test_results.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
            print("\n📄 Test results saved to 'test_results.json'")
            
            return True
            
    except Exception as e:
        print(f"❌ Quick trip planning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 AI TRAVEL AGENT - SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Environment Configuration", test_environment),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Quick Trip Planning", test_quick_trip_planning)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Open ai_travel_agent_demo.ipynb for interactive examples")
        print("2. Run: python ai_travel_agent.py for command-line usage")
        print("3. Check the README.md for detailed documentation")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all API keys are set in .env file")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check your internet connection for API access")
    
    return passed == total

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "env":
            test_environment()
        elif test_type == "imports":
            test_imports()
        elif test_type == "basic":
            test_basic_functionality()
        elif test_type == "quick":
            test_quick_trip_planning()
        else:
            print("Available test types: env, imports, basic, quick, all")
    else:
        run_all_tests()

if __name__ == "__main__":
    main()