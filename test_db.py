from langchain_helper import get_few_shot_db_chain

def test_database_connection():
    try:
        print("Testing database connection...")
        chain = get_few_shot_db_chain()
        print("✓ Database connection successful!")
        print("✓ Chain created successfully!")
        
        # Test with a simple question
        print("\nTesting with a sample question...")
        test_question = "How many t-shirts do we have?"
        result = chain.invoke({"input": test_question})
        print(f"✓ Query executed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Check specific error types
        if "Access denied" in str(e):
            print("\n🔐 Database access denied:")
            print("- Check username/password in langchain_helper.py")
            print("- Current credentials: root/J21042003@127.0.0.1")
            
        elif "Unknown database" in str(e):
            print("\n🗃️ Database doesn't exist:")
            print("- Create 'atliq_tshirts' database in MySQL")
            print("- Import your table structure and data")
            
        elif "Can't connect" in str(e):
            print("\n🔌 Connection failed:")
            print("- Make sure MySQL server is running")
            print("- Check if port 3306 is accessible")
            
        else:
            print(f"\n❓ Other error: {type(e).__name__}")

if __name__ == "__main__":
    test_database_connection()
