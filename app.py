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
    Instead, return: SELECT 'Hello! How can I help you with notices or routines today?' AS message;
- Return only the SQL query, nothing else (no explanations, formatting, or comments)
- Use proper SQL syntax for MySQL
- Always include LIMIT 5 at the end of the query
- Only access the tables: api_notice and api_routine
- Never access or expose author IDs or other sensitive fields
- Do not use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or schema-altering statements
- If the question cannot be answered using api_notice or api_routine, return:
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
    Ensure query is read-only and safe.
    """
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    return not any(keyword in query.upper() for keyword in forbidden)


def execute_sql(query: str):
    """
    Run the SQL safely with error handling.
    """
    if not is_safe_sql(query):
        return {"error": "Unsafe SQL detected. Only SELECT queries are allowed."}

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

    # Execute SQL
    execution_result = execute_sql(cleaned_sql)

    # If error, return directly
    if "error" in execution_result or "message" in execution_result:
        return cleaned_sql, execution_result

    # Convert DB result to natural language
    nl_response = llm.invoke(
        f"SQL: {cleaned_sql}\nResult: {execution_result['result']}\n"
        f"Convert this into a concise natural language answer without SQL, IDs, or technical terms."
    )

    return cleaned_sql, nl_response
