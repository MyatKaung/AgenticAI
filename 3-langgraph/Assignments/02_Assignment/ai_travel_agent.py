#!/usr/bin/env python3
"""
AI Travel Agent & Expense Planner
A comprehensive travel planning system using LangGraph and LangChain
with real-time data integration and expense calculation.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from enum import Enum
import requests

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
# Removed PydanticOutputParser due to Pydantic v2 compatibility issues
# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = "llama3-70b-8192"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048

# Data Models
@dataclass
class TripRequest:
    destination: str
    start_date: str
    end_date: str
    budget: float
    currency: str
    travelers: int
    preferences: List[str]

@dataclass
class WeatherInfo:
    current_temp: float
    condition: str
    forecast: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class Attraction:
    name: str
    description: str
    rating: float
    price: float
    category: str
    location: str

@dataclass
class Hotel:
    name: str
    price_per_night: float
    rating: float
    amenities: List[str]
    location: str

@dataclass
class ExpenseBreakdown:
    accommodation: float
    food: float
    transportation: float
    activities: float
    miscellaneous: float
    total: float
    daily_budget: float

@dataclass
class Itinerary:
    day: int
    date: str
    activities: List[Dict[str, Any]]
    meals: List[Dict[str, Any]]
    estimated_cost: float

class TravelState(TypedDict):
    trip_request: Dict[str, Any]
    weather_info: Dict[str, Any]
    attractions: List[Dict[str, Any]]
    hotels: List[Dict[str, Any]]
    expenses: Dict[str, Any]
    currency_rates: Dict[str, Any]
    itinerary: List[Dict[str, Any]]
    summary: str
    current_step: str
    errors: List[str]

# Base Agent Class
class BaseAgent(ABC):
    def __init__(self, llm: ChatGroq, tools: List[Any] = None):
        self.llm = llm
        self.tools = tools or []
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, state: TravelState) -> TravelState:
        pass
    
    def _search_web(self, query: str, search_tool: str = "tavily") -> str:
        """Perform web search using specified tool"""
        try:
            if search_tool == "tavily" and Config.TAVILY_API_KEY:
                tavily = TavilySearchResults(api_key=Config.TAVILY_API_KEY, max_results=5)
                results = tavily.run(query)
            else:
                ddg = DuckDuckGoSearchRun()
                results = ddg.run(query)
            return str(results)
        except Exception as e:
            return f"Search error: {str(e)}"

# Specialized Agent Classes
class WeatherAgent(BaseAgent):
    """Agent responsible for weather information and forecasting using Open-Meteo API"""
    
    def _get_coordinates(self, destination: str) -> tuple:
        """Get latitude and longitude for a destination using web search"""
        try:
            coords_query = f"{destination} latitude longitude coordinates"
            coords_data = self._search_web(coords_query)
            
            # Use LLM to extract coordinates
            coords_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract latitude and longitude coordinates from the given text."),
                ("human", f"""Extract the latitude and longitude for {destination} from this data:
                {coords_data}
                
                Return only a JSON object with keys 'lat' and 'lon' as numbers.
                Example: {{"lat": 40.7128, "lon": -74.0060}}""")
            ])
            
            response = self.llm.invoke(coords_prompt.format_messages())
            try:
                coords = json.loads(response.content)
                return coords.get('lat', 0), coords.get('lon', 0)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in coordinates: {e}")
                print(f"LLM response was: {response.content}")
                # Default coordinates (New York) if extraction fails
                return 40.7128, -74.0060
        except Exception as e:
            print(f"Error getting coordinates: {e}")
            # Default coordinates (New York) if extraction fails
            return 40.7128, -74.0060
    
    def _fetch_weather_data(self, lat: float, lon: float) -> dict:
        """Fetch weather data from Open-Meteo API"""
        try:
            # Current weather and 7-day forecast
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return {}
    
    def _interpret_weather_code(self, code: int) -> str:
        """Convert WMO weather code to description"""
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown")
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        destination = trip_request["destination"]
        
        try:
            # Get coordinates for the destination
            lat, lon = self._get_coordinates(destination)
            
            # Fetch weather data from Open-Meteo
            weather_data = self._fetch_weather_data(lat, lon)
            
            if weather_data:
                current = weather_data.get('current', {})
                daily = weather_data.get('daily', {})
                
                # Process current weather
                current_temp = current.get('temperature_2m', 'N/A')
                current_condition = self._interpret_weather_code(current.get('weather_code', 0))
                
                # Process forecast
                forecast = []
                if daily.get('time'):
                    for i, date in enumerate(daily['time'][:7]):
                        forecast.append({
                            "date": date,
                            "condition": self._interpret_weather_code(daily['weather_code'][i]),
                            "max_temp": daily['temperature_2m_max'][i],
                            "min_temp": daily['temperature_2m_min'][i],
                            "precipitation": daily['precipitation_sum'][i]
                        })
                
                # Generate travel recommendations
                recommendations = []
                if current_temp != 'N/A':
                    if current_temp < 10:
                        recommendations.append("Pack warm clothing and layers")
                    elif current_temp > 25:
                        recommendations.append("Pack light, breathable clothing and sun protection")
                    else:
                        recommendations.append("Comfortable weather for outdoor activities")
                
                if any(day.get('precipitation', 0) > 5 for day in forecast):
                    recommendations.append("Pack rain gear and plan indoor activities")
                
                weather_info = {
                    "current_temp": f"{current_temp}°C" if current_temp != 'N/A' else "N/A",
                    "condition": current_condition,
                    "forecast": forecast,
                    "recommendations": recommendations or ["Check weather before departure"]
                }
            else:
                weather_info = {
                    "current_temp": "N/A",
                    "condition": "Data unavailable",
                    "forecast": [],
                    "recommendations": ["Check weather before departure"]
                }
                
        except Exception as e:
            print(f"Weather agent error: {e}")
            weather_info = {
                "current_temp": "N/A",
                "condition": "Data unavailable",
                "forecast": [],
                "recommendations": ["Check weather before departure"]
            }
        
        state["weather_info"] = weather_info
        state["current_step"] = "weather_completed"
        return state

class AttractionAgent(BaseAgent):
    """Agent responsible for finding attractions, activities, and restaurants"""
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        destination = trip_request["destination"]
        preferences = trip_request.get("preferences", [])
        
        # Search for attractions
        attractions_query = f"top attractions activities {destination} tourist places visit"
        attractions_data = self._search_web(attractions_query)
        
        # Search for restaurants
        restaurants_query = f"best restaurants {destination} local food dining"
        restaurants_data = self._search_web(restaurants_query)
        
        # Search for activities based on preferences
        activities_query = f"{destination} activities {' '.join(preferences)} things to do"
        activities_data = self._search_web(activities_query)
        
        # Process with LLM
        attractions_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a travel expert specializing in attractions and activities."),
            ("human", f"""Analyze and structure this information for {destination}:
            Attractions: {attractions_data}
            Restaurants: {restaurants_data}
            Activities: {activities_data}
            
            Create a structured list of:
            1. Top 10 attractions with ratings, descriptions, and estimated costs
            2. Top 10 restaurants with cuisine types and price ranges
            3. Top 10 activities matching preferences: {preferences}
            
            Format as JSON array with objects containing: name, description, rating, price, category, location""")
        ])
        
        try:
            response = self.llm.invoke(attractions_prompt.format_messages())
            attractions = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in attractions: {e}")
            print(f"LLM response was: {response.content}")
            attractions = [{
                "name": "Local Exploration",
                "description": "Explore the local area",
                "rating": 4.0,
                "price": 0,
                "category": "sightseeing",
                "location": destination
            }]
        except Exception as e:
            print(f"Error in attractions agent: {e}")
            attractions = [{
                "name": "Local Exploration",
                "description": "Explore the local area",
                "rating": 4.0,
                "price": 0,
                "category": "sightseeing",
                "location": destination
            }]
        
        state["attractions"] = attractions
        state["current_step"] = "attractions_completed"
        return state

class HotelAgent(BaseAgent):
    """Agent responsible for hotel search and cost estimation"""
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        destination = trip_request["destination"]
        budget = trip_request["budget"]
        travelers = trip_request["travelers"]
        
        # Search for hotels
        hotels_query = f"hotels {destination} accommodation booking prices per night {travelers} guests"
        hotels_data = self._search_web(hotels_query)
        
        # Search for budget options
        budget_query = f"budget hotels {destination} cheap accommodation under {budget//7} per night"
        budget_data = self._search_web(budget_query)
        
        # Process with LLM
        hotels_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a hotel booking expert."),
            ("human", f"""Find and structure hotel information for {destination}:
            Hotels data: {hotels_data}
            Budget options: {budget_data}
            
            Budget: ${budget} total
            Travelers: {travelers}
            
            Provide 10 hotel options across different price ranges with:
            - Hotel name
            - Price per night
            - Rating
            - Key amenities
            - Location
            
            Format as JSON array with objects containing: name, price_per_night, rating, amenities, location""")
        ])
        
        try:
            response = self.llm.invoke(hotels_prompt.format_messages())
            hotels = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in hotels: {e}")
            print(f"LLM response was: {response.content}")
            hotels = [{
                "name": "Budget Hotel",
                "price_per_night": budget // 14,  # Half budget for accommodation
                "rating": 3.5,
                "amenities": ["WiFi", "Breakfast"],
                "location": destination
            }]
        except Exception as e:
            print(f"Error in hotels agent: {e}")
            hotels = [{
                "name": "Budget Hotel",
                "price_per_night": budget // 14,  # Half budget for accommodation
                "rating": 3.5,
                "amenities": ["WiFi", "Breakfast"],
                "location": destination
            }]
        
        state["hotels"] = hotels
        state["current_step"] = "hotels_completed"
        return state

class CostCalculatorAgent(BaseAgent):
    """Agent responsible for calculating total expenses and daily budgets"""
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        hotels = state.get("hotels", [])
        attractions = state.get("attractions", [])
        
        start_date = datetime.strptime(trip_request["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(trip_request["end_date"], "%Y-%m-%d")
        days = (end_date - start_date).days
        travelers = trip_request["travelers"]
        
        # Calculate accommodation costs
        avg_hotel_price = sum(h["price_per_night"] for h in hotels[:3]) / min(3, len(hotels)) if hotels else 100
        accommodation_cost = avg_hotel_price * days
        
        # Estimate other costs
        food_cost = 50 * days * travelers  # $50 per person per day
        transportation_cost = 200 * travelers  # Fixed transportation cost
        
        # Calculate activity costs
        activity_costs = [a.get("price", 0) for a in attractions if a.get("price", 0) > 0]
        avg_activity_cost = sum(activity_costs[:10]) / max(1, len(activity_costs[:10]))
        activities_cost = avg_activity_cost * days * 0.7  # Assume 70% of activities are paid
        
        miscellaneous_cost = (accommodation_cost + food_cost + activities_cost) * 0.1  # 10% buffer
        
        total_cost = accommodation_cost + food_cost + transportation_cost + activities_cost + miscellaneous_cost
        daily_budget = total_cost / days
        
        expenses = {
            "accommodation": accommodation_cost,
            "food": food_cost,
            "transportation": transportation_cost,
            "activities": activities_cost,
            "miscellaneous": miscellaneous_cost,
            "total": total_cost,
            "daily_budget": daily_budget,
            "days": days
        }
        
        state["expenses"] = expenses
        state["current_step"] = "costs_completed"
        return state

class CurrencyAgent(BaseAgent):
    """Agent responsible for currency conversion"""
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        expenses = state.get("expenses", {})
        target_currency = trip_request["currency"]
        
        if target_currency.upper() == "USD":
            # No conversion needed
            state["currency_rates"] = {"USD": 1.0}
            return state
        
        # Search for exchange rates
        exchange_query = f"USD to {target_currency} exchange rate current"
        exchange_data = self._search_web(exchange_query)
        
        # Process with LLM
        currency_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a currency exchange expert."),
            ("human", f"""Extract the current exchange rate from this data:
            {exchange_data}
            
            Find the rate to convert USD to {target_currency}.
            Return only the numeric rate as a JSON object: {{"rate": number}}""")
        ])
        
        try:
            response = self.llm.invoke(currency_prompt.format_messages())
            rate_data = json.loads(response.content)
            exchange_rate = rate_data.get("rate", 1.0)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in currency: {e}")
            print(f"LLM response was: {response.content}")
            exchange_rate = 1.0  # Default to 1:1 if conversion fails
        except Exception as e:
            print(f"Error in currency agent: {e}")
            exchange_rate = 1.0  # Default to 1:1 if conversion fails
        
        # Convert all expenses
        converted_expenses = {}
        for key, value in expenses.items():
            if isinstance(value, (int, float)):
                converted_expenses[key] = value * exchange_rate
            else:
                converted_expenses[key] = value
        
        state["currency_rates"] = {target_currency: exchange_rate}
        state["expenses"] = converted_expenses
        state["current_step"] = "currency_completed"
        return state

class ItineraryAgent(BaseAgent):
    """Agent responsible for generating complete itinerary"""
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        attractions = state.get("attractions", [])
        weather_info = state.get("weather_info", {})
        expenses = state.get("expenses", {})
        
        start_date = datetime.strptime(trip_request["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(trip_request["end_date"], "%Y-%m-%d")
        days = (end_date - start_date).days
        
        # Generate day-by-day itinerary
        itinerary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert travel itinerary planner."),
            ("human", f"""Create a detailed {days}-day itinerary for {trip_request['destination']}:
            
            Available attractions: {json.dumps(attractions[:15])}
            Weather info: {json.dumps(weather_info)}
            Daily budget: ${expenses.get('daily_budget', 200):.2f}
            Travelers: {trip_request['travelers']}
            Preferences: {trip_request.get('preferences', [])}
            
            Create a day-by-day plan with:
            - Morning, afternoon, and evening activities
            - Meal recommendations
            - Transportation suggestions
            - Estimated daily costs
            - Weather considerations
            
            Format as JSON array with objects containing:
            day, date, activities (array), meals (array), estimated_cost""")
        ])
        
        try:
            response = self.llm.invoke(itinerary_prompt.format_messages())
            itinerary = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in itinerary: {e}")
            print(f"LLM response was: {response.content}")
            # Generate basic itinerary
            itinerary = []
            for i in range(days):
                current_date = start_date + timedelta(days=i)
                day_plan = {
                    "day": i + 1,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "activities": [
                        {"time": "Morning", "activity": "Explore local attractions", "cost": 30},
                        {"time": "Afternoon", "activity": "Visit museums or landmarks", "cost": 25},
                        {"time": "Evening", "activity": "Dinner and local entertainment", "cost": 45}
                    ],
                    "meals": [
                        {"meal": "Breakfast", "suggestion": "Local cafe", "cost": 15},
                        {"meal": "Lunch", "suggestion": "Street food", "cost": 20},
                        {"meal": "Dinner", "suggestion": "Traditional restaurant", "cost": 35}
                    ],
                    "estimated_cost": 170
                }
                itinerary.append(day_plan)
        except Exception as e:
            print(f"Error in itinerary agent: {e}")
            # Generate basic itinerary
            itinerary = []
            for i in range(days):
                current_date = start_date + timedelta(days=i)
                day_plan = {
                    "day": i + 1,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "activities": [
                        {"time": "Morning", "activity": "Explore local attractions", "cost": 30},
                        {"time": "Afternoon", "activity": "Visit museums or landmarks", "cost": 25},
                        {"time": "Evening", "activity": "Dinner and local entertainment", "cost": 45}
                    ],
                    "meals": [
                        {"meal": "Breakfast", "suggestion": "Local cafe", "cost": 15},
                        {"meal": "Lunch", "suggestion": "Street food", "cost": 20},
                        {"meal": "Dinner", "suggestion": "Traditional restaurant", "cost": 35}
                    ],
                    "estimated_cost": 170
                }
                itinerary.append(day_plan)
        
        state["itinerary"] = itinerary
        state["current_step"] = "itinerary_completed"
        return state

class SummaryAgent(BaseAgent):
    """Agent responsible for generating final trip summary"""
    
    def execute(self, state: TravelState) -> TravelState:
        trip_request = state["trip_request"]
        weather_info = state.get("weather_info", {})
        attractions = state.get("attractions", [])
        hotels = state.get("hotels", [])
        expenses = state.get("expenses", {})
        itinerary = state.get("itinerary", [])
        currency = trip_request["currency"]
        
        # Generate comprehensive summary with safe JSON serialization
        try:
            weather_str = json.dumps(weather_info, ensure_ascii=False)
            attractions_str = json.dumps(attractions[:5], ensure_ascii=False)
            hotels_str = json.dumps(hotels[:3], ensure_ascii=False)
            expenses_str = json.dumps(expenses, ensure_ascii=False)
            itinerary_str = json.dumps(itinerary, ensure_ascii=False)
        except Exception as e:
            # Fallback to string representation if JSON serialization fails
            weather_str = str(weather_info)
            attractions_str = str(attractions[:5])
            hotels_str = str(hotels[:3])
            expenses_str = str(expenses)
            itinerary_str = str(itinerary)
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional travel consultant creating a comprehensive trip summary."),
            ("human", f"""Create a detailed travel plan summary for:
            
            Destination: {trip_request['destination']}
            Dates: {trip_request['start_date']} to {trip_request['end_date']}
            Travelers: {trip_request['travelers']}
            Budget: {trip_request['budget']} {currency}
            
            Weather: {weather_str}
            Top Attractions: {attractions_str}
            Recommended Hotels: {hotels_str}
            Expense Breakdown: {expenses_str}
            Itinerary: {itinerary_str}
            
            Create a professional, comprehensive travel plan summary including:
            1. Trip Overview
            2. Weather Forecast & Recommendations
            3. Top Attractions & Activities
            4. Accommodation Recommendations
            5. Detailed Expense Breakdown
            6. Day-by-Day Itinerary
            7. Travel Tips & Recommendations
            8. Total Cost Summary
            
            Format as a well-structured markdown document.""")
        ])
        
        try:
            response = self.llm.invoke(summary_prompt.format_messages())
            summary = response.content
        except Exception as e:
            print(f"Summary generation error: {e}")
            print(f"Weather data being processed: {weather_info}")
            summary = f"""# Travel Plan Summary
            
            **Destination:** {trip_request['destination']}
            **Dates:** {trip_request['start_date']} to {trip_request['end_date']}
            **Travelers:** {trip_request['travelers']}
            **Total Budget:** {expenses.get('total', 0):.2f} {currency}
            
            ## Expense Breakdown
            - Accommodation: {expenses.get('accommodation', 0):.2f} {currency}
            - Food: {expenses.get('food', 0):.2f} {currency}
            - Transportation: {expenses.get('transportation', 0):.2f} {currency}
            - Activities: {expenses.get('activities', 0):.2f} {currency}
            - Miscellaneous: {expenses.get('miscellaneous', 0):.2f} {currency}
            
            ## Daily Budget
            {expenses.get('daily_budget', 0):.2f} {currency} per day
            
            Note: Detailed summary generation encountered an issue. Basic summary provided.
            """
        
        state["summary"] = summary
        state["current_step"] = "completed"
        return state

# Supervisor Agent
class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates the workflow"""
    
    def __init__(self, llm: ChatGroq):
        super().__init__(llm)
        self.agents = {
            "weather": WeatherAgent(llm),
            "attractions": AttractionAgent(llm),
            "hotels": HotelAgent(llm),
            "costs": CostCalculatorAgent(llm),
            "currency": CurrencyAgent(llm),
            "itinerary": ItineraryAgent(llm),
            "summary": SummaryAgent(llm)
        }
    
    def route_next(self, state: TravelState) -> str:
        """Determine the next agent to execute"""
        current_step = state.get("current_step", "start")
        
        routing_map = {
            "start": "weather",
            "weather_completed": "attractions",
            "attractions_completed": "hotels",
            "hotels_completed": "costs",
            "costs_completed": "currency",
            "currency_completed": "itinerary",
            "itinerary_completed": "summary",
            "completed": END
        }
        
        return routing_map.get(current_step, END)
    
    def execute(self, state: TravelState) -> TravelState:
        """Execute the appropriate agent based on current state"""
        next_agent = self.route_next(state)
        
        if next_agent == END:
            return state
        
        if next_agent in self.agents:
            return self.agents[next_agent].execute(state)
        
        return state

# Main Travel Agent System
class AITravelAgent:
    """Main AI Travel Agent system using LangGraph"""
    
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.supervisor = SupervisorAgent(self.llm)
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(TravelState)
        
        # Add nodes
        workflow.add_node("supervisor", self.supervisor.execute)
        workflow.add_node("weather_agent", self.supervisor.agents["weather"].execute)
        workflow.add_node("attractions_agent", self.supervisor.agents["attractions"].execute)
        workflow.add_node("hotels_agent", self.supervisor.agents["hotels"].execute)
        workflow.add_node("costs_agent", self.supervisor.agents["costs"].execute)
        workflow.add_node("currency_agent", self.supervisor.agents["currency"].execute)
        workflow.add_node("itinerary_agent", self.supervisor.agents["itinerary"].execute)
        workflow.add_node("summary_agent", self.supervisor.agents["summary"].execute)
        
        # Add edges
        workflow.set_entry_point("weather_agent")
        workflow.add_edge("weather_agent", "attractions_agent")
        workflow.add_edge("attractions_agent", "hotels_agent")
        workflow.add_edge("hotels_agent", "costs_agent")
        workflow.add_edge("costs_agent", "currency_agent")
        workflow.add_edge("currency_agent", "itinerary_agent")
        workflow.add_edge("itinerary_agent", "summary_agent")
        workflow.add_edge("summary_agent", END)
        
        return workflow.compile()
    
    def plan_trip(self, trip_request: TripRequest) -> Dict[str, Any]:
        """Plan a complete trip based on the request"""
        initial_state = TravelState(
            trip_request={
                "destination": trip_request.destination,
                "start_date": trip_request.start_date,
                "end_date": trip_request.end_date,
                "budget": trip_request.budget,
                "currency": trip_request.currency,
                "travelers": trip_request.travelers,
                "preferences": trip_request.preferences
            },
            weather_info={},
            attractions=[],
            hotels=[],
            expenses={},
            currency_rates={},
            itinerary=[],
            summary="",
            current_step="start",
            errors=[]
        )
        
        try:
            # Execute the workflow
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            return {
                "error": f"Trip planning failed: {str(e)}",
                "partial_result": initial_state
            }

# Example usage and testing
def main():
    """Example usage of the AI Travel Agent"""
    # Initialize the travel agent
    agent = AITravelAgent()
    
    # Create a sample trip request
    trip_request = TripRequest(
        destination="Paris, France",
        start_date="2025-07-19",
        end_date="2025-07-22",
        budget=3000.0,
        currency="USD",
        travelers=2,
        preferences=["museums", "restaurants", "historical sites", "art galleries"]
    )
    
    print("🌍 AI Travel Agent & Expense Planner")
    print("=" * 50)
    print(f"Planning trip to: {trip_request.destination}")
    print(f"Dates: {trip_request.start_date} to {trip_request.end_date}")
    print(f"Budget: ${trip_request.budget} {trip_request.currency}")
    print(f"Travelers: {trip_request.travelers}")
    print("\nProcessing...\n")
    
    # Plan the trip
    result = agent.plan_trip(trip_request)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display results
    print("✅ Trip planning completed!")
    print("\n" + "=" * 50)
    print(result.get("summary", "Summary not available"))
    
    # Log success instead of saving to JSON file
    print("\n📋 Trip planning process completed successfully!")
    print(f"📊 Total estimated cost: ${result.get('expenses', {}).get('total', 'N/A')}")
    print(f"📅 Daily budget: ${result.get('expenses', {}).get('daily_budget', 'N/A'):.2f}" if result.get('expenses', {}).get('daily_budget') else "📅 Daily budget: N/A")

if __name__ == "__main__":
    main()