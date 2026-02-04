# ChatBot-using-Agents

Built a ChatBot for PartSelect website using Agents. 
- Scraped the PartSelect website using Selenium.
- Saved the data into Postgres Database.
- Used LangGraph to orchestrate an agentic flow: Analyse -> Retrieve -> Generate.
- Exposed the agent through FastAPI, and responses are streamed using Server-Sent Events, so the UI feels like a real chatbot.
- Developed the UI using React and node.js.
