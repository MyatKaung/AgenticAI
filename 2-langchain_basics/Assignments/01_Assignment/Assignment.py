import os 
from dotenv import load_dotenv
load_dotenv()
os.getenv("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

#Langsmith Tracking and Tracing 
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser # Import StrOutputParser
from langchain_groq import ChatGroq # Assuming you're using ChatGroq as in your working example

# 1. Define Pydantic model for the product
class Product(BaseModel):
    product_name: str = Field(description="The name of the product")
    product_details: str = Field(description="Detailed description of the product")
    tentative_price_usd: int = Field(description="The tentative price of the product in USD")

# 2. Set up the PydanticOutputParser
pydantic_parser = PydanticOutputParser(pydantic_object=Product)

# 3. Create the ChatPromptTemplate 
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert assistant. Your task is to extract product information "
        "from the user's text and format it as a JSON object according to the schema below. "
        "Only output the JSON object. If the user's text does not contain enough information "
        "for a field, you should try to infer it or use a sensible placeholder. "
        "Prioritize accuracy based on the input. Ensure the output strictly follows the provided JSON schema."
        "\n\nSchema:\n{format_instructions}"
    )),
    ("human", "{user_query}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())

if __name__ == "__main__":
    try:
        llm = ChatGroq(model="gemma2-9b-it", temperature=0.7)
        print("Using ChatGroq(model='gemma2-9b-it', temperature=0.7)")
        print("Make sure your GROQ_API_KEY environment variable is set if using Groq.")

    except ImportError:
        print("ImportError: langchain_groq is not installed.")
        print("Please install it: pip install langchain-groq")
        exit()
    except Exception as e:
        print(f"Error instantiating ChatGroq: {e}")
        print("This could be due to a missing API key or other configuration issues.")
        exit()

    # Chain to get the RAW STRING output from LLM (checked whether it is JSON))
    chain_for_raw_llm_output = prompt_template | llm | StrOutputParser()

    # Original chain that parses into Pydantic object
    chain_for_pydantic_object = prompt_template | llm | pydantic_parser

    # query1
    query1 = "Tell me about the Apple iPhone 16 Pro Max."

    print(f"\nProcessing Query: {query1}")

    # --- Get and print the raw LLM string output ---
    print("\n--- Attempting to get RAW LLM String Output (should be JSON) ---")
    try:
        raw_output_string = chain_for_raw_llm_output.invoke({"user_query": query1})
        print("Raw output string from LLM:")
        print(raw_output_string)
    except Exception as e:
        print(f"Error getting raw LLM output: {e}")

    # --- Get and print the parsed Pydantic object ---
    print("\n--- Attempting to parse into Pydantic Object ---")
    try:
        product_info = chain_for_pydantic_object.invoke({"user_query": query1})
        print("Successfully parsed into Pydantic Product object:")
        print(f"  Python Object Representation: {product_info}") 
        print(f"  As JSON from Pydantic object: {product_info.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error processing into Pydantic object: {e}")
