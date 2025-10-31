import os
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Database config
db_user = os.environ.get("DB_USER")
db_password = os.environ.get("DB_PASS")
db_host = os.environ.get("DB_HOST")
db_name = os.environ.get("DB_NAME")

mysql_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
db = SQLDatabase.from_uri(mysql_uri)

# Template with stricter rules
template = """Based on the following SQL database schema, generate ONLY a valid SQL query to answer the user's question.

IMPORTANT RULES:
- If the user greets (e.g., "hi", "hello", "hey", "good morning", "how are you"), do not generate SQL.
    Instead, return: SELECT 'Hello! How can I help you today?' AS message;
- Return only the SQL query, nothing else (no explanations, formatting, or comments)
- Use proper SQL syntax for MySQL
- If the user specifies a number (e.g., "10", "15"), use that as the LIMIT.
- If the user says "more than 5", use LIMIT 10.
- If the user says "all" or does not specify a number, use LIMIT 5 by default.
- Only access the tables: api_notice, api_event and api_routine
- Never access or expose author IDs or other sensitive fields
- Do not use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or schema-altering statements
- If the question cannot be answered using api_notice, api_event or api_routine, return:
  SELECT 'Query not possible based on given rules' AS message;
- If an invalid or unsafe operation is attempted, return:
  SELECT 'Invalid operation. Only SELECT queries are allowed' AS message;

Schema:
{schema}

Question: {question}

SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)


def get_schema(_):
    return db.get_table_info()


# LLM config
llm = GoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    top_p=1,
    top_k=1,
)

# SQL Chain
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()
)


def clean_sql(raw_response: str) -> str:
    """
    Clean up SQL query returned by the LLM.
    Removes code fences and ensures no extra formatting.
    """
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("sql", "", 1).strip()
    return cleaned


def is_safe_sql(query: str) -> bool:
    """
    Ensures only SELECT queries are executed and no unsafe keywords exist.
    """
    # Trim whitespace and semicolons
    query = query.strip().rstrip(";").upper()

    # Allow only queries starting with SELECT
    if not query.startswith("SELECT"):
        return False

    # Check for dangerous commands anywhere in the text
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    return not any(keyword in query for keyword in forbidden)


def execute_sql(query: str):
    """
    Run the SQL safely with error handling.
    """
    if not is_safe_sql(query):
        return {"error": "You are not allowed to perform sensitive operations."}

    try:
        result = db.run(query)
        if not result:
            return {"message": "No results found"}
        return {"result": result}
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}


def get_and_send(question: str):
    """
    Main pipeline: question → SQL → execution → response
    """
    # Generate SQL
    raw_response = sql_chain.invoke({"question": question.strip()})
    cleaned_sql = clean_sql(raw_response)
    print("Raw Query:",raw_response)
    print("Generated SQL:", cleaned_sql)

    # Execute SQL
    execution_result = execute_sql(cleaned_sql)

    # If error, return directly
    if "error" in execution_result:
        # friendly_message = "Sorry, something went wrong. Please try again"
        return cleaned_sql, execution_result["error"]
    if "message" in execution_result:
        return cleaned_sql, execution_result["message"]
    
    # Convert DB result to natural language
    nl_response = llm.invoke(
        f"SQL: {cleaned_sql}\nResult: {execution_result['result']}\n"
        f"Convert this into a concise natural language answer without SQL, IDs, or technical terms."
    )

    return cleaned_sql, nl_response
