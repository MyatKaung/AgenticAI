#!/usr/bin/env python3
"""
AI Travel Agent & Expense Planner
A comprehensive travel planning system using LangGraph and LangChain
with real-time data integration and expense calculation.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict
from abc import ABC, abstractmethod
import re
import logging

# Configure logging to provide detailed insight into the agent's operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Third-party imports
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = "llama3-70b-8192"
    TEMPERATURE = 0.0  # Set to 0 for deterministic and reliable JSON output
    MAX_TOKENS = 4096  # Increased to handle complex itinerary and summary generation

# --- Data Models ---
class TravelState(TypedDict):
    trip_request: Dict[str, Any]
    weather_info: Dict[str, Any]
    attractions: List[Dict[str, Any]]
    hotels: List[Dict[str, Any]]
    expenses: Dict[str, Any]
    itinerary: List[Dict[str, Any]]
    summary: str
    errors: List[str]

# --- Helper Functions ---
def _parse_llm_json(response_content: str, expected_type: type = dict) -> Any:
    """
    Robustly parses JSON from an LLM response by finding the start of the JSON structure
    and cleaning it of any surrounding text or markdown.
    """
    logger = logging.getLogger("_parse_llm_json")
    logger.debug(f"Attempting to parse LLM response (first 500 chars): {response_content[:500]}")
    
    json_start = -1
    obj_start = response_content.find('{')
    arr_start = response_content.find('[')

    if obj_start != -1 and arr_start != -1:
        json_start = min(obj_start, arr_start)
    elif obj_start != -1:
        json_start = obj_start
    elif arr_start != -1:
        json_start = arr_start

    if json_start == -1:
        raise json.JSONDecodeError("No JSON object or array found in the response.", response_content, 0)

    json_str = response_content[json_start:]
    try:
        parsed_json = json.loads(json_str)
        if not isinstance(parsed_json, expected_type):
            raise ValueError(f"Parsed JSON is not of the expected type {expected_type.__name__}.")
        logger.info("Successfully parsed JSON from LLM response.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}. Failed on content: '''{json_str}'''")
        raise e

# --- Base Agent Class ---
class BaseAgent(ABC):
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def execute(self, state: TravelState) -> TravelState:
        pass

    def _search_web(self, query: str) -> str:
        """
        Performs a web search and returns a clean, concatenated string of content.
        """
        self.logger.info(f"Executing web search for query: '{query}'")
        try:
            tavily = TavilySearchResults(api_key=Config.TAVILY_API_KEY, max_results=3)
            results = tavily.invoke(query)
            
            if not results:
                self.logger.warning("Web search returned no results.")
                return "No information found."
            
            content = "\n\n".join([item['content'] for item in results if isinstance(item, dict) and 'content' in item])
            
            self.logger.info(f"Web search returned {len(content)} characters of clean content.")
            return content
        except Exception as e:
            self.logger.error(f"Web search failed for query '{query}': {str(e)}")
            return f"Search error: {str(e)}"

# --- Specialized Agent Classes ---

class WeatherAgent(BaseAgent):
    def execute(self, state: TravelState) -> TravelState:
        self.logger.info("Executing...")
        req = state["trip_request"]
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={req['destination']}&format=json"
            geo_response = requests.get(geo_url, headers={'User-Agent': 'AITravelAgent/1.0'}).json()
            lat, lon = float(geo_response[0]['lat']), float(geo_response[0]['lon'])
        except Exception as e:
            self.logger.error(f"Could not get coordinates for {req['destination']}: {e}. Defaulting.")
            lat, lon = 48.8566, 2.3522 # Default to Paris

        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code", "daily": "weather_code,temperature_2m_max,temperature_2m_min", "timezone": "auto", "forecast_days": 7}
            weather_data = requests.get(url, params=params, timeout=10).json()
            daily = weather_data['daily']
            state["weather_info"] = {
                "current_temp": f"{weather_data.get('current', {}).get('temperature_2m', 'N/A')}°C",
                "forecast": [{"date": daily['time'][i], "max_temp": daily['temperature_2m_max'][i]} for i in range(len(daily['time']))],
                "recommendations": ["Pack layers for changing weather.", "An umbrella is recommended."]
            }
        except Exception as e:
            self.logger.error(f"Weather agent failed: {e}. Using fallback data.")
            state["weather_info"] = {"current_temp": "N/A", "forecast": [], "recommendations": ["Weather data unavailable."]}
            state["errors"].append(f"{self.name}: Failed to retrieve data.")
        self.logger.info("Completed.")
        return state

class AttractionAgent(BaseAgent):
    def execute(self, state: TravelState) -> TravelState:
        self.logger.info("Executing...")
        req = state["trip_request"]
        query = f"Top 5 tourist attractions and restaurants in {req['destination']} for travelers interested in {', '.join(req['preferences'])}."
        search_results = self._search_web(query)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a JSON extractor. Respond with ONLY the JSON array. Do not add any commentary."),
            ("human", "Based on the text below, create a JSON array of 5 attractions and restaurants. Each object MUST have 'name' (string), 'description' (string), 'price' (float, 0 if free), and 'category' ('attraction' or 'restaurant').\n\n{search_results}")
        ])

        try:
            # **CRITICAL FIX**: Chain the prompt and LLM, then invoke with data.
            chain = prompt | self.llm
            response = chain.invoke({"search_results": search_results})
            state['attractions'] = _parse_llm_json(response.content, list)
        except Exception as e:
            self.logger.error(f"Agent failed: {e}. Using fallback data.")
            state['attractions'] = [{"name": "Eiffel Tower", "description": "Iconic landmark of Paris.", "price": 28.0, "category": "attraction"}]
            state["errors"].append(f"{self.name}: Failed to parse LLM response.")
        self.logger.info("Completed.")
        return state

class HotelAgent(BaseAgent):
    def execute(self, state: TravelState) -> TravelState:
        self.logger.info("Executing...")
        req = state["trip_request"]
        query = f"Recommended hotels in {req['destination']} for {req['travelers']} guests."
        search_results = self._search_web(query)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a JSON extractor. Respond with ONLY the JSON array. Do not add any commentary."),
            ("human", "Based on the text below, create a JSON array of 3 hotels. Each object MUST have 'name' (string) and 'price_per_night' (float).\n\n{search_results}")
        ])
        
        try:
            # **CRITICAL FIX**: Chain the prompt and LLM, then invoke with data.
            chain = prompt | self.llm
            response = chain.invoke({"search_results": search_results})
            state['hotels'] = _parse_llm_json(response.content, list)
        except Exception as e:
            self.logger.error(f"Agent failed: {e}. Using fallback data.")
            state['hotels'] = [{"name": "The Ritz Paris", "price_per_night": 2000.0}]
            state["errors"].append(f"{self.name}: Failed to parse LLM response.")
        self.logger.info("Completed.")
        return state

class CostCalculatorAgent(BaseAgent):
    def execute(self, state: TravelState) -> TravelState:
        self.logger.info("Executing...")
        req = state["trip_request"]
        days = max(1, (datetime.strptime(req["end_date"], "%Y-%m-%d") - datetime.strptime(req["start_date"], "%Y-%m-%d")).days)
        
        hotel_prices = [h.get('price_per_night', 200.0) for h in state['hotels'] if isinstance(h.get('price_per_night'), (int, float))]
        avg_hotel_price = sum(hotel_prices) / len(hotel_prices) if hotel_prices else 200.0
        accommodation_cost = avg_hotel_price * days

        food_cost = 80.0 * days * req['travelers']
        transport_cost = 30.0 * days * req['travelers']
        activities_cost = sum(a.get('price', 25.0) for a in state['attractions']) * req['travelers']
        
        total_cost = accommodation_cost + food_cost + transport_cost + activities_cost
        
        state["expenses"] = {
            "accommodation": round(accommodation_cost, 2),
            "food": round(food_cost, 2),
            "transportation": round(transport_cost, 2),
            "activities": round(activities_cost, 2),
            "total": round(total_cost, 2),
            "days": days,  # **CRITICAL FIX**: Ensure 'days' is in the dictionary
            "daily_budget": round(total_cost / days, 2)
        }
        self.logger.info("Completed.")
        return state

class ItineraryAgent(BaseAgent):
    def execute(self, state: TravelState) -> TravelState:
        self.logger.info("Executing...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a JSON itinerary planner. Respond with ONLY the JSON array."),
            ("human", """Create a detailed itinerary for a {days}-day trip to {destination}.
- Use the following attractions: {attractions}
- Consider the daily budget of {daily_budget} {currency}.
Format as a JSON array, where each object represents a day and includes 'day' (int), 'date' (string), 'activities' (list of strings), and 'meals' (list of strings).""")
        ])
        
        try:
            # **CRITICAL FIX**: Chain the prompt and LLM, then invoke with data.
            chain = prompt | self.llm
            response = chain.invoke({
                "days": state['expenses']['days'],
                "destination": state['trip_request']['destination'],
                "attractions": json.dumps(state['attractions']),
                "daily_budget": state['expenses']['daily_budget'],
                "currency": state['trip_request']['currency']
            })
            state['itinerary'] = _parse_llm_json(response.content, list)
        except Exception as e:
            self.logger.error(f"Agent failed: {e}. Using fallback data.")
            state['itinerary'] = [{"day": 1, "date": state['trip_request']['start_date'], "activities": ["Arrival and explore."], "meals": ["Dinner at a local bistro."]}]
            state["errors"].append(f"{self.name}: Failed to parse LLM response.")
        self.logger.info("Completed.")
        return state

class SummaryAgent(BaseAgent):
    def execute(self, state: TravelState) -> TravelState:
        self.logger.info("Executing...")
        
        itinerary_str = ""
        for day in state.get('itinerary', []):
            itinerary_str += f"\n**Day {day.get('day')} ({day.get('date')})**\n"
            itinerary_str += "- Activities: " + ", ".join(day.get('activities', [])) + "\n"
            itinerary_str += "- Meals: " + ", ".join(day.get('meals', [])) + "\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional travel consultant. Generate a markdown summary."),
            ("human", """Create a comprehensive travel plan summary in markdown format.

**Trip Overview**
- Destination: {destination}
- Dates: {start_date} to {end_date}
- Travelers: {travelers}

**Expense Breakdown ({currency})**
- Accommodation: {accommodation:.2f}
- Food: {food:.2f}
- Transportation: {transportation:.2f}
- Activities: {activities:.2f}
- **Total Estimated Cost:** {total:.2f}
- **Daily Budget:** {daily_budget:.2f}

**Suggested Itinerary**
{itinerary_str}

Please provide a polished, final summary based on this data.""")
        ])
        
        try:
            # **CRITICAL FIX**: Chain the prompt and LLM, then invoke with data.
            chain = prompt | self.llm
            response = chain.invoke({
                **state['trip_request'],
                **state['expenses'],
                "itinerary_str": itinerary_str
            })
            state['summary'] = response.content
        except Exception as e:
            self.logger.error(f"Agent failed: {e}. Generating a fallback summary.")
            state['summary'] = "# Travel Plan Summary (Error)\nAn error occurred while generating the detailed summary."
            state["errors"].append(f"{self.name}: Failed to generate summary.")
        self.logger.info("Completed.")
        return state

# --- Main Travel Agent System ---
class AITravelAgent:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(TravelState)
        
        workflow.add_node("weather_node", WeatherAgent(self.llm).execute)
        workflow.add_node("attractions_node", AttractionAgent(self.llm).execute)
        workflow.add_node("hotels_node", HotelAgent(self.llm).execute)
        workflow.add_node("costs_node", CostCalculatorAgent(self.llm).execute)
        workflow.add_node("itinerary_node", ItineraryAgent(self.llm).execute)
        workflow.add_node("summary_node", SummaryAgent(self.llm).execute)

        workflow.set_entry_point("weather_node")
        workflow.add_edge("weather_node", "attractions_node")
        workflow.add_edge("attractions_node", "hotels_node")
        workflow.add_edge("hotels_node", "costs_node")
        workflow.add_edge("costs_node", "itinerary_node")
        workflow.add_edge("itinerary_node", "summary_node")
        workflow.add_edge("summary_node", END)

        return workflow.compile()

    def plan_trip(self, trip_request: dict) -> Dict[str, Any]:
        initial_state = TravelState(
            trip_request=trip_request,
            weather_info={}, attractions=[], hotels=[], expenses={},
            itinerary=[], summary="", errors=[]
        )
        final_state = self.workflow.invoke(initial_state)
        return final_state

# --- Example Usage ---
def main():
    agent = AITravelAgent()
    
    trip_request = {
        "destination": "Paris, France",
        "start_date": "2025-07-19",
        "end_date": "2025-07-22",
        "budget": 3000.0,
        "currency": "USD",
        "travelers": 2,
        "preferences": ["museums", "art", "fine dining", "history"]
    }
    
    print("🌍 AI Travel Agent & Expense Planner")
    print("=" * 50)
    print(f"Planning trip to: {trip_request['destination']}")
    
    result = agent.plan_trip(trip_request)
    
    print("\n✅ Trip planning process completed!")
    if result.get("errors"):
        print("\n⚠️  Errors occurred during planning:")
        for error in result["errors"]:
            print(f"- {error}")
    
    print("\n" + "=" * 50)
    print("Travel Plan Summary")
    print("=" * 50)
    print(result.get("summary", "Summary not available."))

if __name__ == "__main__":
    main()