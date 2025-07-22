# 🛍️ AI-Powered Retail Q&A System

An intelligent database query system that converts natural language questions into SQL queries using Google Gemini AI and LangChain, enabling non-technical users to extract business insights from retail inventory data.

## 🚀 Features

- **Natural Language to SQL**: Ask questions in plain English and get SQL results
- **AI-Powered**: Uses Google Gemini 1.5 Flash for intelligent query generation
- **Semantic Search**: RAG implementation with ChromaDB vector database
- **Clean Results**: Automatically formats database outputs to user-friendly numbers
- **Web Interface**: Simple Streamlit-based UI for easy interaction

## 🛠️ Tech Stack

- **AI/ML**: Google Gemini 1.5 Flash, HuggingFace Transformers, LangChain
- **Database**: MySQL, ChromaDB (Vector Database)
- **Backend**: Python, PyMySQL, SQLAlchemy
- **Frontend**: Streamlit
- **Libraries**: sentence-transformers, pandas, decimal

## 📋 Prerequisites

- Python 3.8+
- MySQL Server
- Google AI API Key

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/retail-qa-system.git
   cd retail-qa-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   API_KEY=your_google_ai_api_key_here
   ```

4. **Set up MySQL Database**
   - Create a database named `atliq_tshirts`
   - Update database credentials in `langchain_helper.py`:
     ```python
     db_user = "your_username"
     db_password = "your_password"
     db_host = "localhost"
     ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 🎯 Usage

1. Start the Streamlit application
2. Open your browser to `http://localhost:8501`
3. Type natural language questions like:
   - "How many Nike t-shirts do we have in size XS?"
   - "What's the total value of our white t-shirt inventory?"
   - "How many t-shirts are discounted above 10%?"

## 🏗️ Project Architecture

```
├── main.py                 # Streamlit web interface
├── langchain_helper.py     # Core AI logic and database integration
├── few_shots.py           # Example Q&A pairs for training
├── test_db.py             # Database testing utilities
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
└── README.md             # Project documentation
```

## 🧠 How It Works

1. **User Input**: Natural language question via Streamlit interface
2. **Semantic Search**: Find relevant examples using vector similarity
3. **Prompt Engineering**: Combine examples with database schema
4. **AI Generation**: Google Gemini generates SQL query
5. **Execution**: Run query on MySQL database
6. **Formatting**: Clean and format results for user display

## 🔧 Key Technologies

- **RAG (Retrieval-Augmented Generation)**: Improves query accuracy with relevant examples
- **Vector Embeddings**: Semantic similarity matching with sentence-transformers
- **Few-Shot Learning**: AI learns from provided examples
- **Prompt Engineering**: Structured prompts for consistent SQL generation

## 📊 Example Queries

| Question | Generated SQL | Result |
|----------|---------------|--------|
| "How many white Nike t-shirts?" | `SELECT SUM(stock_quantity) FROM t_shirts WHERE brand='Nike' AND color='white'` | 25 |
| "Total inventory value for size L?" | `SELECT SUM(price * stock_quantity) FROM t_shirts WHERE size='L'` | 15750 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google AI for Gemini API
- HuggingFace for transformer models
- LangChain for AI application framework
- Streamlit for web interface

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/your-username/retail-qa-system](https://github.com/your-username/retail-qa-system)
