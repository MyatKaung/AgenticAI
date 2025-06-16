#!/usr/bin/env python3
"""
Gradio Web App for AI Travel Agent & Expense Planner
Interactive interface for planning trips with real-time data integration.
"""

import gradio as gr
import json
import os
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    return missing_keys

def load_travel_agent():
    """Load the AI Travel Agent with error handling"""
    try:
        from ai_travel_agent import AITravelAgent, TripRequest
        return AITravelAgent(), TripRequest, None
    except ImportError as e:
        return None, None, f"Import Error: {str(e)}. Please run 'pip install -r requirements.txt'"
    except Exception as e:
        return None, None, f"Error loading travel agent: {str(e)}"

# Global variables
agent, TripRequest, load_error = load_travel_agent()
missing_keys = check_environment()

def plan_trip_interface(
    destination,
    start_date,
    end_date,
    budget,
    currency,
    travelers,
    preferences,
    custom_preferences
):
    """Main trip planning function for Gradio interface"""
    
    # Environment checks
    if missing_keys:
        return (
            f"❌ **Configuration Error**\n\nMissing API keys: {', '.join(missing_keys)}\n\nPlease set up your .env file with the required API keys.",
            "",
            "",
            "",
            "",
            ""
        )
    
    if load_error:
        return (
            f"❌ **System Error**\n\n{load_error}",
            "",
            "",
            "",
            "",
            ""
        )
    
    # Input validation
    if not destination or destination.strip() == "":
        return (
            "❌ **Input Error**\n\nPlease enter a destination.",
            "",
            "",
            "",
            "",
            ""
        )
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_dt >= end_dt:
            return (
                "❌ **Date Error**\n\nEnd date must be after start date.",
                "",
                "",
                "",
                "",
                ""
            )
        
        if start_dt < date.today():
            return (
                "❌ **Date Error**\n\nStart date cannot be in the past.",
                "",
                "",
                "",
                "",
                ""
            )
    
    except ValueError:
        return (
            "❌ **Date Error**\n\nInvalid date format. Please use YYYY-MM-DD.",
            "",
            "",
            "",
            "",
            ""
        )
    
    # Process preferences
    all_preferences = []
    if preferences:
        all_preferences.extend(preferences)
    
    if custom_preferences and custom_preferences.strip():
        custom_prefs = [pref.strip() for pref in custom_preferences.split(",") if pref.strip()]
        all_preferences.extend(custom_prefs)
    
    if not all_preferences:
        return (
            "❌ **Preference Error**\n\nPlease select at least one preference or add custom preferences.",
            "",
            "",
            "",
            "",
            ""
        )
    
    try:
        # Create trip request
        trip_request = TripRequest(
            destination=destination.strip(),
            start_date=start_date,
            end_date=end_date,
            budget=float(budget),
            currency=currency,
            travelers=int(travelers),
            preferences=all_preferences
        )
        
        # Plan the trip
        result = agent.plan_trip(trip_request)
        
        if "error" in result:
            return (
                f"❌ **Trip Planning Error**\n\n{result['error']}",
                "",
                "",
                "",
                "",
                ""
            )
        
        # Format results
        trip_info = format_trip_info(trip_request, result)
        expense_breakdown = format_expense_breakdown(result.get("expenses", {}))
        weather_info = format_weather_info(result.get("weather_info", {}))
        attractions_info = format_attractions(result.get("attractions", []))
        itinerary_info = format_itinerary(result.get("itinerary", []))
        json_output = json.dumps(result, indent=2, default=str)
        
        return (
            trip_info,
            expense_breakdown,
            weather_info,
            attractions_info,
            itinerary_info,
            json_output
        )
    
    except Exception as e:
        error_msg = f"❌ **Unexpected Error**\n\n{str(e)}\n\nPlease check your API keys and internet connection."
        if "debug" in str(e).lower():
            error_msg += f"\n\n**Debug Information:**\n```\n{traceback.format_exc()}\n```"
        
        return (
            error_msg,
            "",
            "",
            "",
            "",
            ""
        )

def format_trip_info(trip_request, result):
    """Format basic trip information"""
    duration = (datetime.strptime(trip_request.end_date, "%Y-%m-%d") - 
                datetime.strptime(trip_request.start_date, "%Y-%m-%d")).days
    
    info = f"""# ✈️ Trip Plan Summary

**📍 Destination:** {trip_request.destination}
**📅 Duration:** {trip_request.start_date} to {trip_request.end_date} ({duration} days)
**💰 Budget:** {trip_request.budget:,.2f} {trip_request.currency}
**👥 Travelers:** {trip_request.travelers}
**🎯 Preferences:** {', '.join(trip_request.preferences[:5])}{'...' if len(trip_request.preferences) > 5 else ''}

---

## 📋 Trip Summary

{result.get('summary', 'Summary not available')}
"""
    
    return info

def format_expense_breakdown(expenses):
    """Format expense breakdown"""
    if not expenses:
        return "💰 **Expense information not available**"
    
    total = expenses.get('total', 0)
    daily_budget = expenses.get('daily_budget', 0)
    
    breakdown = f"""# 💰 Expense Breakdown

## 💸 Total Cost: ${total:,.2f}
## 📅 Daily Budget: ${daily_budget:,.2f}

---

### Detailed Breakdown:

🏨 **Accommodation:** ${expenses.get('accommodation', 0):,.2f}
🍽️ **Food & Dining:** ${expenses.get('food', 0):,.2f}
🚗 **Transportation:** ${expenses.get('transportation', 0):,.2f}
🎭 **Activities:** ${expenses.get('activities', 0):,.2f}
💼 **Miscellaneous:** ${expenses.get('miscellaneous', 0):,.2f}
"""
    
    # Add percentage breakdown if total > 0
    if total > 0:
        breakdown += "\n### Percentage Breakdown:\n\n"
        categories = [
            ('accommodation', '🏨 Accommodation'),
            ('food', '🍽️ Food'),
            ('transportation', '🚗 Transportation'),
            ('activities', '🎭 Activities'),
            ('miscellaneous', '💼 Miscellaneous')
        ]
        
        for key, label in categories:
            amount = expenses.get(key, 0)
            percentage = (amount / total) * 100 if total > 0 else 0
            breakdown += f"{label}: {percentage:.1f}%\n"
    
    return breakdown

def format_weather_info(weather):
    """Format weather information"""
    if not weather:
        return "🌤️ **Weather information not available**"
    
    info = "# 🌤️ Weather Information\n\n"
    
    if weather.get('current'):
        info += f"**Current Weather:** {weather['current']}\n\n"
    
    if weather.get('forecast'):
        info += f"**Forecast:** {weather['forecast']}\n\n"
    
    if weather.get('recommendations'):
        info += f"**Recommendations:** {weather['recommendations']}\n\n"
    
    return info

def format_attractions(attractions):
    """Format attractions list"""
    if not attractions:
        return "🏛️ **Attraction information not available**"
    
    info = "# 🏛️ Recommended Attractions\n\n"
    
    for i, attraction in enumerate(attractions[:10], 1):
        info += f"{i}. {attraction}\n"
    
    if len(attractions) > 10:
        info += f"\n... and {len(attractions) - 10} more attractions\n"
    
    return info

def format_itinerary(itinerary):
    """Format daily itinerary"""
    if not itinerary:
        return "📅 **Itinerary not available**"
    
    info = "# 📅 Daily Itinerary\n\n"
    
    for day in itinerary:
        day_title = day.get('date', 'Day') + (f" - {day.get('title', '')}" if day.get('title') else "")
        info += f"## {day_title}\n\n"
        
        activities = day.get('activities', 'No activities planned')
        info += f"{activities}\n\n---\n\n"
    
    return info

def get_example_trip(destination_example):
    """Get example trip data based on destination"""
    examples = {
        "Paris, France": {
            "start_date": (date.today() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": (date.today() + timedelta(days=37)).strftime("%Y-%m-%d"),
            "budget": 3000.0,
            "currency": "USD",
            "travelers": 2,
            "preferences": ["museums", "art galleries", "restaurants", "historical sites"]
        },
        "Tokyo, Japan": {
            "start_date": (date.today() + timedelta(days=45)).strftime("%Y-%m-%d"),
            "end_date": (date.today() + timedelta(days=52)).strftime("%Y-%m-%d"),
            "budget": 4000.0,
            "currency": "USD",
            "travelers": 1,
            "preferences": ["temples", "technology", "food", "traditional culture"]
        },
        "New York, USA": {
            "start_date": (date.today() + timedelta(days=60)).strftime("%Y-%m-%d"),
            "end_date": (date.today() + timedelta(days=65)).strftime("%Y-%m-%d"),
            "budget": 2500.0,
            "currency": "USD",
            "travelers": 2,
            "preferences": ["museums", "theater", "restaurants", "shopping"]
        },
        "Rome, Italy": {
            "start_date": (date.today() + timedelta(days=40)).strftime("%Y-%m-%d"),
            "end_date": (date.today() + timedelta(days=46)).strftime("%Y-%m-%d"),
            "budget": 2800.0,
            "currency": "EUR",
            "travelers": 2,
            "preferences": ["historical sites", "architecture", "restaurants", "art galleries"]
        },
        "London, UK": {
            "start_date": (date.today() + timedelta(days=35)).strftime("%Y-%m-%d"),
            "end_date": (date.today() + timedelta(days=42)).strftime("%Y-%m-%d"),
            "budget": 3200.0,
            "currency": "GBP",
            "travelers": 2,
            "preferences": ["museums", "theater", "historical sites", "pubs"]
        }
    }
    
    example = examples.get(destination_example, examples["Paris, France"])
    
    return (
        destination_example,
        example["start_date"],
        example["end_date"],
        example["budget"],
        example["currency"],
        example["travelers"],
        example["preferences"],
        ""
    )

# Define preference options
preference_options = [
    "museums", "art galleries", "historical sites", "architecture",
    "restaurants", "local cuisine", "street food", "fine dining",
    "shopping", "nightlife", "bars", "clubs",
    "nature", "parks", "beaches", "hiking",
    "adventure", "sports", "outdoor activities",
    "culture", "festivals", "music", "theater",
    "technology", "science", "modern attractions",
    "temples", "churches", "spiritual sites",
    "family-friendly", "kids activities",
    "photography", "scenic views", "landmarks"
]

# Create Gradio interface
with gr.Blocks(
    title="AI Travel Agent & Expense Planner",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    """
) as app:
    
    gr.HTML("""
    <div class="main-header">
        <h1>✈️ AI Travel Agent & Expense Planner</h1>
        <p>Plan your perfect trip with real-time data and AI-powered recommendations</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 🌍 Plan Your Trip")
            
            # Quick examples
            gr.Markdown("### 🎯 Quick Examples")
            example_dropdown = gr.Dropdown(
                choices=["Paris, France", "Tokyo, Japan", "New York, USA", "Rome, Italy", "London, UK"],
                label="Select an example destination",
                value=None
            )
            
            # Trip details
            destination = gr.Textbox(
                label="🏙️ Destination",
                placeholder="e.g., Paris, France or Tokyo, Japan",
                info="Enter the city and country you want to visit"
            )
            
            with gr.Row():
                start_date = gr.Textbox(
                    label="📅 Start Date (YYYY-MM-DD)",
                    value=(date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
                )
                end_date = gr.Textbox(
                    label="📅 End Date (YYYY-MM-DD)",
                    value=(date.today() + timedelta(days=37)).strftime("%Y-%m-%d")
                )
            
            with gr.Row():
                budget = gr.Number(
                    label="💰 Budget",
                    value=3000.0,
                    minimum=100,
                    maximum=50000
                )
                currency = gr.Dropdown(
                    label="💱 Currency",
                    choices=["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"],
                    value="USD"
                )
                travelers = gr.Number(
                    label="👥 Travelers",
                    value=2,
                    minimum=1,
                    maximum=10,
                    precision=0
                )
            
            # Preferences
            gr.Markdown("### 🎯 Travel Preferences")
            preferences = gr.CheckboxGroup(
                label="Select your interests:",
                choices=preference_options,
                value=["museums", "restaurants", "historical sites"]
            )
            
            custom_preferences = gr.Textbox(
                label="✏️ Additional preferences (comma-separated)",
                placeholder="e.g., vegetarian food, budget accommodations, luxury experiences",
                info="Add any specific preferences not listed above"
            )
            
            # Plan trip button
            plan_button = gr.Button(
                "🚀 Plan My Trip",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Features")
            gr.Markdown("""
            - 🌤️ **Real-time Weather** - Current conditions and forecasts
            - 🏛️ **Top Attractions** - Curated activities and sights
            - 🏨 **Hotel Costs** - Accommodation pricing estimates
            - 💱 **Currency Conversion** - Real-time exchange rates
            - 📅 **Daily Itinerary** - Day-by-day planning
            - 💰 **Expense Breakdown** - Detailed cost analysis
            - 📋 **Trip Summary** - Comprehensive travel report
            """)
            
            gr.Markdown("## ℹ️ About")
            gr.Markdown("""
            This AI Travel Agent uses:
            - **LangGraph** for workflow orchestration
            - **LangChain** for LLM integration
            - **Groq** for fast AI responses
            - **Tavily** for real-time web search
            - **Multi-agent architecture** for specialized tasks
            """)
    
    # Results section
    gr.Markdown("## 📋 Trip Planning Results")
    
    with gr.Tabs():
        with gr.TabItem("📋 Trip Summary"):
            trip_summary = gr.Markdown()
        
        with gr.TabItem("💰 Expenses"):
            expense_breakdown = gr.Markdown()
        
        with gr.TabItem("🌤️ Weather"):
            weather_info = gr.Markdown()
        
        with gr.TabItem("🏛️ Attractions"):
            attractions_info = gr.Markdown()
        
        with gr.TabItem("📅 Itinerary"):
            itinerary_info = gr.Markdown()
        
        with gr.TabItem("📄 Raw Data (JSON)"):
            json_output = gr.Code(language="json", label="Complete Trip Data")
    
    # Event handlers
    example_dropdown.change(
        fn=get_example_trip,
        inputs=[example_dropdown],
        outputs=[
            destination,
            start_date,
            end_date,
            budget,
            currency,
            travelers,
            preferences,
            custom_preferences
        ]
    )
    
    plan_button.click(
        fn=plan_trip_interface,
        inputs=[
            destination,
            start_date,
            end_date,
            budget,
            currency,
            travelers,
            preferences,
            custom_preferences
        ],
        outputs=[
            trip_summary,
            expense_breakdown,
            weather_info,
            attractions_info,
            itinerary_info,
            json_output
        ]
    )
    
    # Footer
    gr.Markdown("""
    ---
    
    ### 🆘 Support
    
    **Common Issues:**
    - Ensure API keys are set in `.env` file
    - Check internet connection for real-time data
    - Verify all dependencies are installed with `pip install -r requirements.txt`
    
    **Quick Links:**
    - [📖 Documentation](README.md)
    - [🧪 Test Script](test_travel_agent.py)
    - [💻 Source Code](ai_travel_agent.py)
    """)

if __name__ == "__main__":
    # Check environment before launching
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set up your .env file with the required API keys.")
        print("The app will still launch but won't function properly without API keys.")
    
    if load_error:
        print(f"❌ Error loading travel agent: {load_error}")
        print("Please run 'pip install -r requirements.txt' to install dependencies.")
    
    print("🚀 Launching Gradio AI Travel Agent...")
    print("📱 The app will open in your default web browser.")
    
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False
    )