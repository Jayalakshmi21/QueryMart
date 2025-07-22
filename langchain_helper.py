from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
from decimal import Decimal

from few_shots import few_shots

import os
from dotenv import load_dotenv
load_dotenv()  


def get_few_shot_db_chain():
    db_user = os.environ.get("DB_USER", "root")
    db_password = os.environ.get("DB_PASSWORD", "your_password")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_name = os.environ.get("DB_NAME", "atliq_tshirts")

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["API_KEY"], temperature=0.1)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Create a simple list of questions for vectorization
    example_texts = []
    example_metadatas = []
    for i, example in enumerate(few_shots):
        example_texts.append(example["Question"])
        # Flatten the metadata to avoid nested dict issues
        example_metadatas.append({
            "id": i,
            "question": example["Question"],
            "sql_query": example["SQLQuery"],
            "sql_result": example["SQLResult"],
            "answer": str(example["Answer"])
        })
    
    vectorstore = Chroma.from_texts(example_texts, embeddings, metadatas=example_metadatas)
    
    def get_relevant_examples(query: str) -> list:
        docs = vectorstore.similarity_search(query, k=2)
        examples = []
        for doc in docs:
            examples.append({
                "Question": doc.metadata["question"],
                "SQLQuery": doc.metadata["sql_query"],
                "SQLResult": doc.metadata["sql_result"],
                "Answer": doc.metadata["answer"]
            })
        return examples
    
    mysql_prompt = """You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run.
    
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. 
    Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. 
    Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    IMPORTANT: Only return the SQL query, nothing else. Do not include explanations or additional text.
    
    Use these examples as reference for query patterns:"""

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery"],
        template="\nQuestion: {Question}\nSQL: {SQLQuery}",
    )

    # Create a complete chain that generates SQL, executes it, and returns the answer
    def create_prompt_with_examples(input_text: str) -> str:
        examples = get_relevant_examples(input_text)
        
        example_strings = []
        for example in examples:
            # Only include Question and SQLQuery in examples
            example_data = {
                "Question": example["Question"],
                "SQLQuery": example["SQLQuery"]
            }
            example_strings.append(example_prompt.format(**example_data))
        
        examples_text = "\n".join(example_strings)
        
        # Get table info directly here
        table_info = db.get_table_info()
        
        prompt = f"""{mysql_prompt}

{examples_text}

Only use the following tables:
{table_info}

Question: {input_text}
SQL: """
        return prompt

    def execute_sql_chain(input_data):
        question = input_data["input"]
        
        # Step 1: Generate SQL query
        sql_prompt = create_prompt_with_examples(question)
        sql_query = llm.invoke(sql_prompt).content.strip()
        
        # Clean up the SQL query (remove any extra text and markdown)
        # Remove markdown code blocks
        sql_query = re.sub(r'```sql\s*', '', sql_query)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Extract just the SQL part if there's extra formatting
        lines = sql_query.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    sql_query = line
                    break
        
        # Remove any remaining prefixes
        if sql_query.startswith('SQL:'):
            sql_query = sql_query[4:].strip()
            
        try:
            # Step 2: Execute the SQL query
            result = db.run(sql_query)
            
            # Step 3: Format the final answer - return clean numbers only
            if result:
                if isinstance(result, str) and result.strip():
                    # Check if it's a string representation of a list/tuple with Decimal
                    if result.startswith("[(Decimal(") and result.endswith(")]"):
                        # Extract the number from string like "[(Decimal('6961.000000'),)]"
                        match = re.search(r"Decimal\('([^']+)'\)", result)
                        if match:
                            decimal_str = match.group(1)
                            try:
                                numeric_value = float(decimal_str)
                                if numeric_value == int(numeric_value):
                                    return str(int(numeric_value))
                                else:
                                    return str(numeric_value).rstrip('0').rstrip('.')
                            except (ValueError, TypeError):
                                return decimal_str
                    else:
                        return result.strip()
                elif isinstance(result, list) and len(result) > 0:
                    first_result = result[0]
                    if isinstance(first_result, tuple) and len(first_result) > 0:
                        value = first_result[0]
                    else:
                        value = first_result
                    
                    # Clean formatting - extract just the number
                    if value is None:
                        return "No data found"
                    else:
                        # Handle Decimal type specifically
                        if isinstance(value, Decimal):
                            # Convert Decimal to string, then to float to remove trailing zeros
                            numeric_value = float(str(value))
                            if numeric_value == int(numeric_value):
                                return str(int(numeric_value))
                            else:
                                return str(numeric_value).rstrip('0').rstrip('.')
                        elif isinstance(value, (int, float)):
                            # Handle regular numbers
                            if isinstance(value, float) and value == int(value):
                                return str(int(value))
                            else:
                                return str(value).rstrip('0').rstrip('.')
                        else:
                            # Try to extract number from string representation
                            value_str = str(value)
                            try:
                                # Try to convert to float first
                                numeric_value = float(value_str)
                                if numeric_value == int(numeric_value):
                                    return str(int(numeric_value))
                                else:
                                    return str(numeric_value).rstrip('0').rstrip('.')
                            except (ValueError, TypeError):
                                # Return the value as is if can't convert
                                return value_str
                else:
                    return "No data found"
            else:
                return "No data found"
                
        except Exception as e:
            # Check if it's a database structure issue
            if "doesn't exist" in str(e).lower():
                return f"Database table doesn't exist. Please ensure the 'atliq_tshirts' database is set up with the required tables.\nGenerated SQL: {sql_query}"
            elif "access denied" in str(e).lower():
                return f"Database access denied. Please check your database credentials.\nGenerated SQL: {sql_query}"
            else:
                return f"Database error: {str(e)}\nGenerated SQL: {sql_query}"
    
    # Create a chain that generates SQL, executes it, and returns the final answer
    chain = RunnablePassthrough() | execute_sql_chain
    
    return chain