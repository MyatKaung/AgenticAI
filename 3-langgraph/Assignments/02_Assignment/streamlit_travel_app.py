"""
Streamlit Web App for AI Travel Agent & Expense Planner
Interactive interface for planning trips with real-time data integration.
"""

import streamlit as st
import json
import os
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Travel Agent & Expense Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.expense-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

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

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ AI Travel Agent & Expense Planner</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Environment check
        missing_keys = check_environment()
        if missing_keys:
            st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
            st.info("Please set up your .env file with the required API keys.")
            st.stop()
        else:
            st.success("✅ Environment configured")
        
        # Load travel agent
        agent, TripRequest, error = load_travel_agent()
        if error:
            st.error(f"❌ {error}")
            st.stop()
        else:
            st.success("✅ AI Travel Agent loaded")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🌍 Plan Your Trip")
        
        # Trip planning form
        with st.form("trip_form"):
            # Destination
            destination = st.text_input(
                "🏙️ Destination",
                placeholder="e.g., Paris, France or Tokyo, Japan",
                help="Enter the city and country you want to visit"
            )
            
            # Dates
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "📅 Start Date",
                    value=date.today() + timedelta(days=30),
                    min_value=date.today()
                )
            with col_date2:
                end_date = st.date_input(
                    "📅 End Date",
                    value=date.today() + timedelta(days=37),
                    min_value=start_date if start_date else date.today()
                )
            
            # Budget and travelers
            col_budget1, col_budget2, col_budget3 = st.columns(3)
            with col_budget1:
                budget = st.number_input(
                    "💰 Budget",
                    min_value=100.0,
                    max_value=50000.0,
                    value=3000.0,
                    step=100.0
                )
            with col_budget2:
                currency = st.selectbox(
                    "💱 Currency",
                    ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
                )
            with col_budget3:
                travelers = st.number_input(
                    "👥 Travelers",
                    min_value=1,
                    max_value=10,
                    value=2
                )
            
            # Preferences
            st.subheader("🎯 Travel Preferences")
            preferences = st.multiselect(
                "Select your interests:",
                [
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
                ],
                default=["museums", "restaurants", "historical sites"]
            )
            
            # Custom preferences
            custom_prefs = st.text_input(
                "✏️ Additional preferences (optional)",
                placeholder="e.g., vegetarian food, budget accommodations, luxury experiences"
            )
            
            if custom_prefs:
                preferences.extend([pref.strip() for pref in custom_prefs.split(",")])
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Plan My Trip",
                type="primary",
                use_container_width=True
            )
        
        # Process trip planning
        if submitted:
            if not destination:
                st.error("❌ Please enter a destination")
            elif start_date >= end_date:
                st.error("❌ End date must be after start date")
            elif not preferences:
                st.error("❌ Please select at least one preference")
            else:
                # Create trip request
                trip_request = TripRequest(
                    destination=destination,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    budget=float(budget),
                    currency=currency,
                    travelers=int(travelers),
                    preferences=preferences
                )
                
                # Display trip details
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.write("**🎯 Trip Request Created:**")
                st.write(f"📍 **Destination:** {destination}")
                st.write(f"📅 **Duration:** {start_date} to {end_date} ({(end_date - start_date).days} days)")
                st.write(f"💰 **Budget:** {budget:,.2f} {currency}")
                st.write(f"👥 **Travelers:** {travelers}")
                st.write(f"🎯 **Preferences:** {', '.join(preferences[:5])}{'...' if len(preferences) > 5 else ''}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Plan the trip
                with st.spinner("🔄 Planning your trip... This may take 2-3 minutes for real-time data..."):
                    try:
                        result = agent.plan_trip(trip_request)
                        
                        if "error" in result:
                            st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {result["error"]}</div>', unsafe_allow_html=True)
                        else:
                            st.success("✅ Trip planning completed successfully!")
                            
                            # Store result in session state
                            st.session_state.trip_result = result
                            st.session_state.trip_request = trip_request
                            
                            # Display results
                            display_trip_results(result)
                    
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ <strong>Unexpected Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
                        st.error("Please check your API keys and internet connection.")
                        with st.expander("🔍 Debug Information"):
                            st.code(traceback.format_exc())
    
    with col2:
        st.header("📊 Features")
        
        features = [
            ("🌤️", "Real-time Weather", "Current conditions and forecasts"),
            ("🏛️", "Top Attractions", "Curated activities and sights"),
            ("🏨", "Hotel Costs", "Accommodation pricing estimates"),
            ("💱", "Currency Conversion", "Real-time exchange rates"),
            ("📅", "Daily Itinerary", "Day-by-day planning"),
            ("💰", "Expense Breakdown", "Detailed cost analysis"),
            ("📋", "Trip Summary", "Comprehensive travel report")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-box">
                <h4>{icon} {title}</h4>
                <p style="margin: 0; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick examples
        st.header("🎯 Quick Examples")
        
        examples = [
            ("🗼", "Paris, France", "Art, cuisine, romance"),
            ("🗾", "Tokyo, Japan", "Technology, culture, food"),
            ("🗽", "New York, USA", "Museums, Broadway, dining"),
            ("🏛️", "Rome, Italy", "History, architecture, food"),
            ("🌉", "London, UK", "History, theater, pubs")
        ]
        
        for icon, city, theme in examples:
            if st.button(f"{icon} {city}", key=city, use_container_width=True):
                st.info(f"💡 Try: {city} with preferences like {theme}")

def display_trip_results(result):
    """Display the trip planning results"""
    st.header("🎉 Your Trip Plan")
    
    # Expense summary
    expenses = result.get("expenses", {})
    if expenses:
        st.subheader("💰 Expense Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="expense-card">
                <h3>🏨 Hotels</h3>
                <h2>${expenses.get('accommodation', 0):,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="expense-card">
                <h3>🍽️ Food</h3>
                <h2>${expenses.get('food', 0):,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="expense-card">
                <h3>🚗 Transport</h3>
                <h2>${expenses.get('transportation', 0):,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="expense-card">
                <h3>🎭 Activities</h3>
                <h2>${expenses.get('activities', 0):,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Total cost
        total_cost = expenses.get('total', 0)
        daily_budget = expenses.get('daily_budget', 0)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
            <h2>💸 Total Cost: ${total_cost:,.2f}</h2>
            <h4>📅 Daily Budget: ${daily_budget:,.2f}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Trip summary
    summary = result.get("summary", "")
    if summary:
        st.subheader("📋 Trip Summary")
        st.markdown(summary)
    
    # Weather information
    weather = result.get("weather_info", {})
    if weather:
        st.subheader("🌤️ Weather Information")
        st.info(f"**Current Weather:** {weather.get('current', 'Information not available')}")
        if weather.get('forecast'):
            st.write(f"**Forecast:** {weather['forecast']}")
    
    # Attractions
    attractions = result.get("attractions", [])
    if attractions:
        st.subheader("🏛️ Recommended Attractions")
        for i, attraction in enumerate(attractions[:5], 1):
            st.write(f"{i}. {attraction}")
    
    # Itinerary
    itinerary = result.get("itinerary", [])
    if itinerary:
        st.subheader("📅 Daily Itinerary")
        for day in itinerary[:3]:  # Show first 3 days
            with st.expander(f"📅 {day.get('date', 'Day')} - {day.get('title', 'Activities')}"):
                st.write(day.get('activities', 'No activities planned'))
    
    # Download results
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        json_data = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"trip_plan_{result.get('trip_request', {}).get('destination', 'unknown').replace(', ', '_').replace(' ', '_')}.json",
            mime="application/json"
        )
    
    with col2:
        # Summary download
        summary_text = result.get("summary", "Trip summary not available")
        st.download_button(
            label="📝 Download Summary",
            data=summary_text,
            file_name=f"trip_summary_{result.get('trip_request', {}).get('destination', 'unknown').replace(', ', '_').replace(' ', '_')}.md",
            mime="text/markdown"
        )

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This AI Travel Agent uses:
    - **LangGraph** for workflow orchestration
    - **LangChain** for LLM integration
    - **Groq** for fast AI responses
    - **Tavily** for real-time web search
    - **Multi-agent architecture** for specialized tasks
    """)
    
    st.header("🔗 Quick Links")
    st.markdown("""
    - [📖 Documentation](README.md)
    - [🧪 Test Script](test_travel_agent.py)
    - [💻 Source Code](ai_travel_agent.py)
    """)
    
    st.header("🆘 Support")
    st.markdown("""
    **Common Issues:**
    - Ensure API keys are set in `.env`
    - Check internet connection
    - Verify all dependencies are installed
    """)

if __name__ == "__main__":
    main()