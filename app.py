from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import uuid
from typing import Dict, Optional, List
import requests
from dotenv import load_dotenv
import base64
import warnings
import re
import json
import time
import random

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for API routes

# Configuration
GOOGLE_API_KEY = ""
GITHUB_TOKEN = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Cache repositories to avoid repeated API calls
REPO_CACHE = {}
ISSUE_CACHE = {}
GUIDE_CACHE = {}
CACHE_EXPIRY = 3600  # Cache expires after 1 hour

# Request validation schema
class ChatRequestSchema(Schema):
    conversation_id = fields.Str(required=True)
    question = fields.Str(required=True)
    use_realtime = fields.Bool(missing=True)

chat_request_schema = ChatRequestSchema()

class OpenSourceChat:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )
        self.vectorstore = None
        self.conversation_chain = None
        self.user_preferences = {
            "languages": [],
            "interests": [],
            "previous_repos": []
        }
    
    def initialize_vectorstore(self, curated_data: list[str] = None):
        # Load a much richer set of data about open source contributing
        contributing_guides = [
            "Contributing to open source requires: 1) Finding a project 2) Understanding the codebase 3) Picking an issue 4) Making changes 5) Submitting a PR",
            "Good first issues are typically labeled with 'good first issue', 'beginner friendly', 'easy', or 'help wanted' tags on GitHub",
            "The typical contribution workflow includes: fork the repo, clone locally, create branch, make changes, commit, push, and open a PR",
            "Documentation contributions are excellent for beginners as they help you learn the codebase while making valuable additions",
            "Bug fixes are another good entry point, especially for simple issues that don't require deep knowledge of the codebase",
            "Testing contributions help ensure the project's reliability and are often welcoming to new contributors",
            "Always read the project's README and CONTRIBUTING guides before starting work",
            "Many projects require tests and documentation for new features or bug fixes",
            "Code style and conventions vary by project - look for a style guide or follow the existing patterns",
            "Communication is key - don't hesitate to ask questions in issues or pull requests",
            "Open source etiquette includes being respectful, patient, and open to feedback",
            "Your first PR might not be perfect, and that's okay - the review process is a learning opportunity",
            "Different projects have different review processes and response times - be patient"
        ]
        
        repos_info = [
            "freeCodeCamp/freeCodeCamp: Learn to code for free with millions of learners. Great for beginners with issues spanning various difficulty levels.",
            "firstcontributions/first-contributions: Specifically designed to help beginners make their first contribution with a step-by-step guide.",
            "tensorflow/tensorflow: Machine learning framework with many beginner-friendly issues and excellent documentation.",
            "microsoft/vscode: Popular code editor with active community and well-labeled issues for newcomers.",
            "kubernetes/kubernetes: Container orchestration platform with 'good first issue' labels and comprehensive contributor guides.",
            "flutter/flutter: UI toolkit with many entry-level tasks and supportive community.",
            "rust-lang/rust: Systems programming language with mentored issues for beginners.",
            "home-assistant/core: Home automation platform with various complexity levels of issues.",
            "scikit-learn/scikit-learn: Machine learning library with detailed contribution guidelines and mentor support.",
            "mozilla/firefox-ios: iOS browser with well-documented codebase and beginner issues.",
            "electron/electron: Framework for building cross-platform desktop apps with JS, HTML, and CSS.",
            "NixOS/nixpkgs: Package collection with many simple package updates perfect for first-time contributors.",
            "pandas-dev/pandas: Data analysis library with issues suitable for Python beginners.",
            "react-native-community: Collection of packages supporting React Native with many entry points.",
            "ethereum/ethereum-org-website: Ethereum's website with content and translation tasks ideal for non-developers."
        ]
        
        language_specific = [
            "Python beginners might enjoy contributing to Django, Flask, FastAPI, or Pytest.",
            "JavaScript developers can start with React, Vue.js, or Express projects.",
            "Java contributors can look at Spring Boot or Apache projects.",
            "Ruby beginners often start with Rails or Jekyll.",
            "Go developers can contribute to Docker, Kubernetes, or Hugo.",
            "Rust learners might enjoy working on Rustlings or Rust-Analyzer.",
            "C# developers can contribute to .NET projects or Unity.",
            "PHP contributors can work on Laravel, Symfony, or WordPress.",
            "C/C++ developers might consider SQLite, Redis, or various Linux projects."
        ]
        
        all_data = contributing_guides + repos_info + language_specific
        if curated_data:
            all_data.extend(curated_data)
            
        chunks = self.text_splitter.split_text("\n".join(all_data))
        self.vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={'k': 5}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def _extract_language_preferences(self, question: str) -> List[str]:
        """Extract programming language preferences from user questions"""
        common_languages = [
            "python", "javascript", "typescript", "java", "c#", "c++", "go", "rust", 
            "ruby", "php", "kotlin", "swift", "html", "css", "shell", "scala", "r"
        ]
        
        found_languages = []
        for lang in common_languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', question.lower()):
                found_languages.append(lang)
                
        return found_languages
    
    def _extract_interests(self, question: str) -> List[str]:
        """Extract topic interests from user questions"""
        common_interests = [
            "web", "mobile", "data science", "machine learning", "ai", "game", 
            "database", "frontend", "backend", "fullstack", "devops", "cloud",
            "security", "blockchain", "iot", "embedded", "desktop"
        ]
        
        found_interests = []
        for interest in common_interests:
            if interest in question.lower():
                found_interests.append(interest)
                
        return found_interests
    
    def _update_user_preferences(self, question: str):
        """Update user preferences based on their questions"""
        # Extract languages
        languages = self._extract_language_preferences(question)
        if languages:
            for lang in languages:
                if lang not in self.user_preferences["languages"]:
                    self.user_preferences["languages"].append(lang)
        
        # Extract interests
        interests = self._extract_interests(question)
        if interests:
            for interest in interests:
                if interest not in self.user_preferences["interests"]:
                    self.user_preferences["interests"].append(interest)
    
    def search_repositories(self, query: str = "", language: str = "") -> list[dict]:
        """Search for repositories with improved caching and query building"""
        cache_key = f"{query}_{language}"
        current_time = time.time()
        
        # Check if we have cached results that aren't expired
        if cache_key in REPO_CACHE and current_time - REPO_CACHE[cache_key]["timestamp"] < CACHE_EXPIRY:
            return REPO_CACHE[cache_key]["data"]
        
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        url = "https://api.github.com/search/repositories"
        
        # Build a more effective query
        query_parts = []
        
        # Add the main query
        if query:
            query_parts.append(query)
        
        # Add language if specified
        if language:
            query_parts.append(f"language:{language}")
        # Or use user preferences if available
        elif self.user_preferences["languages"] and not language:
            lang_filter = " OR ".join([f"language:{lang}" for lang in self.user_preferences["languages"][:2]])
            query_parts.append(f"({lang_filter})")
            
        # Always look for beginner-friendly repos
        query_parts.append("(good-first-issues:>0 OR help-wanted-issues:>0)")
        
        # Ensure it's active and somewhat popular
        query_parts.append("stars:>100")
        query_parts.append("pushed:>2023-01-01")
        
        # Make sure they accept PRs
        query_parts.append("is:public fork:false archived:false")
        
        full_query = " ".join(query_parts)
        
        params = {
            "q": full_query,
            "sort": "stars",
            "order": "desc",
            "per_page": 10
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            repos = response.json().get("items", [])
            
            # Process the repositories to add more information
            processed_repos = [
                {
                    "name": repo["full_name"],
                    "description": repo["description"] or "No description available",
                    "url": repo["html_url"],
                    "stars": repo["stargazers_count"],
                    "language": repo["language"] or "Various",
                    "updated_at": repo["updated_at"],
                    "open_issues_count": repo["open_issues_count"],
                    "has_issues": repo["has_issues"]
                }
                for repo in repos
            ]
            
            # Cache the results
            REPO_CACHE[cache_key] = {
                "data": processed_repos,
                "timestamp": current_time
            }
            
            # Update user preferences with the repos
            for repo in processed_repos:
                if repo["name"] not in self.user_preferences["previous_repos"]:
                    self.user_preferences["previous_repos"].append(repo["name"])
                    if len(self.user_preferences["previous_repos"]) > 10:
                        self.user_preferences["previous_repos"].pop(0)
            
            return processed_repos
        except Exception as e:
            # If we get an error, return empty list and log the error
            print(f"GitHub API error: {str(e)}")
            return []

    def search_issues(self, repo_full_name: str) -> list[dict]:
        """Search for issues with improved caching and sorting"""
        cache_key = f"issues_{repo_full_name}"
        current_time = time.time()
        
        # Check if we have cached results that aren't expired
        if cache_key in ISSUE_CACHE and current_time - ISSUE_CACHE[cache_key]["timestamp"] < CACHE_EXPIRY:
            return ISSUE_CACHE[cache_key]["data"]
        
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        # First check if the repo has good first issues
        url = f"https://api.github.com/repos/{repo_full_name}/issues"
        
        # Look for beginner-friendly issues first
        beginner_params = {
            "labels": "good first issue,help wanted,beginner,easy,starter",
            "state": "open",
            "per_page": 10
        }
        
        try:
            response = requests.get(url, headers=headers, params=beginner_params)
            response.raise_for_status()
            beginner_issues = response.json()
            
            # If we don't find enough beginner issues, get some regular issues too
            if len(beginner_issues) < 5:
                regular_params = {
                    "state": "open",
                    "per_page": 5
                }
                regular_response = requests.get(url, headers=headers, params=regular_params)
                regular_response.raise_for_status()
                regular_issues = regular_response.json()
                
                # Combine but avoid duplicates
                beginner_ids = {issue["id"] for issue in beginner_issues}
                combined_issues = beginner_issues + [issue for issue in regular_issues if issue["id"] not in beginner_ids]
            else:
                combined_issues = beginner_issues
            
            # Process the issues
            processed_issues = [
                {
                    "title": issue["title"],
                    "number": issue["number"],
                    "url": issue["html_url"],
                    "labels": [label["name"] for label in issue.get("labels", [])],
                    "created_at": issue["created_at"],
                    "description": (issue["body"][:300] + "..." if issue["body"] and len(issue["body"]) > 300 
                                   else (issue["body"] or "No description available"))
                }
                for issue in combined_issues[:10]  # Limit to 10 issues
            ]
            
            # Cache the results
            ISSUE_CACHE[cache_key] = {
                "data": processed_issues,
                "timestamp": current_time
            }
            
            return processed_issues
        except Exception as e:
            print(f"GitHub API error for issues: {str(e)}")
            return []

    def get_contribution_guide(self, repo_full_name: str) -> str:
        """Get contribution guide with improved caching and fallbacks"""
        cache_key = f"guide_{repo_full_name}"
        current_time = time.time()
        
        # Check if we have cached results that aren't expired
        if cache_key in GUIDE_CACHE and current_time - GUIDE_CACHE[cache_key]["timestamp"] < CACHE_EXPIRY:
            return GUIDE_CACHE[cache_key]["data"]
        
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        # Try different common contribution file paths
        guide_paths = [
            "CONTRIBUTING.md",
            ".github/CONTRIBUTING.md",
            "docs/CONTRIBUTING.md",
            "CONTRIBUTE.md",
            ".github/CONTRIBUTE.md",
            "docs/CONTRIBUTE.md",
            "DEVELOPMENT.md",
            "docs/DEVELOPMENT.md",
        ]
        
        guide_content = "No specific contribution guide found. Here are general steps to contribute:\n\n"
        guide_content += "1. Fork the repository\n"
        guide_content += "2. Clone your fork locally\n"
        guide_content += "3. Create a new branch for your feature or bugfix\n"
        guide_content += "4. Make your changes with clear commit messages\n"
        guide_content += "5. Push to your fork\n"
        guide_content += "6. Submit a pull request to the original repository\n\n"
        guide_content += "Look for issues labeled 'good first issue' or 'help wanted' for beginner-friendly tasks."
        
        for path in guide_paths:
            try:
                url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    content = base64.b64decode(response.json()["content"]).decode("utf-8")
                    
                    # If guide is too long, truncate it
                    if len(content) > 2000:
                        content = content[:2000] + "\n...\n[Guide truncated. See full guide at the repository]"
                    
                    guide_content = f"Contribution guide for {repo_full_name}:\n\n{content}"
                    break
            except Exception:
                continue
        
        # Cache the results
        GUIDE_CACHE[cache_key] = {
            "data": guide_content,
            "timestamp": current_time
        }
        
        return guide_content

    def _extract_repo_from_question(self, question: str) -> str:
        """Extract repository name from question with improved logic"""
        # Check for explicit mentions of repositories in format "username/repo"
        repo_pattern = r'([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)'
        repos = re.findall(repo_pattern, question)
        
        if repos:
            return repos[0]
        
        # Look for "contribute to X" pattern
        contribute_match = re.search(r'contribute to\s+([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)', question, re.IGNORECASE)
        if contribute_match:
            return contribute_match.group(1)
        
        # Look for "issues in X" pattern
        issues_match = re.search(r'issues in\s+([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)', question, re.IGNORECASE)
        if issues_match:
            return issues_match.group(1)
        
        # If we can't find a repo in the question but user has previous repos, use the most recent one
        if self.user_preferences["previous_repos"]:
            return self.user_preferences["previous_repos"][-1]
        
        return ""

    def get_response(self, question: str, use_realtime: bool = True):
        """Process questions and generate responses with dynamic data"""
        # Update user preferences based on the question
        self._update_user_preferences(question)
        
        # Basic categorization of the question
        is_repo_question = any(x in question.lower() for x in ["repository", "repositories", "repos", "projects"])
        is_issue_question = "issue" in question.lower()
        is_contribute_question = any(x in question.lower() for x in ["contribute", "contributing", "contribution"])
        is_guide_question = any(x in question.lower() for x in ["guide", "how to", "steps", "process"])
        
        try:
            # Prepare context data based on question type
            context_data = {}
            
            if is_repo_question:
                # Extract language preference
                language = ""
                if self.user_preferences["languages"]:
                    language = f"language:{self.user_preferences['languages'][0]}"
                
                # Build query based on interests
                query = "good first issue"
                if self.user_preferences["interests"]:
                    query += " " + " ".join(self.user_preferences["interests"][:2])
                
                repos = self.search_repositories(query=query, language=language)
                if repos:
                    repo_list = []
                    for i, repo in enumerate(repos[:5]):  # Limit to 5 repos
                        repo_list.append(f"{i+1}. [{repo['name']}]({repo['url']}): {repo['description']}")
                        repo_list.append(f"   - Stars: {repo['stars']}, Language: {repo['language']}")
                        repo_list.append(f"   - Open Issues: {repo['open_issues_count']}, Last Updated: {repo['updated_at'][:10]}")
                    
                    context_data["repositories"] = "\n".join(repo_list)
                else:
                    context_data["repositories"] = "No repositories found matching your criteria."
            
            # Extract repository name if present
            repo_name = self._extract_repo_from_question(question)
            
            if is_issue_question and repo_name:
                issues = self.search_issues(repo_name)
                if issues:
                    issue_list = []
                    for i, issue in enumerate(issues[:5]):  # Limit to 5 issues
                        issue_list.append(f"{i+1}. [Issue #{issue['number']}]({issue['url']}): {issue['title']}")
                        if issue["labels"]:
                            issue_list.append(f"   - Labels: {', '.join(issue['labels'])}")
                        issue_list.append(f"   - Created: {issue['created_at'][:10]}")
                        issue_list.append(f"   - Description: {issue['description']}")
                    
                    context_data["issues"] = "\n".join(issue_list)
                else:
                    context_data["issues"] = f"No beginner-friendly issues found in {repo_name}."
            
            if (is_contribute_question or is_guide_question) and repo_name:
                guide = self.get_contribution_guide(repo_name)
                context_data["guide"] = guide
            
            # Add user preferences to context
            if self.user_preferences["languages"] or self.user_preferences["interests"]:
                preferences = []
                if self.user_preferences["languages"]:
                    preferences.append(f"Preferred languages: {', '.join(self.user_preferences['languages'])}")
                if self.user_preferences["interests"]:
                    preferences.append(f"Interests: {', '.join(self.user_preferences['interests'])}")
                
                context_data["preferences"] = "\n".join(preferences)
            
            # Build a JSON context that can be passed to the model
            context_json = json.dumps(context_data)
            
            # Enhance the question with the context
            enhanced_question = f"""
            You are GitGuide, an expert assistant for open-source contributors on GitHub. Answer the following question thoughtfully and helpfully.
            
            Use the provided context data to give specific, personalized recommendations. The context includes real-time data from GitHub.
            
            Context data (JSON): {context_json}
            
            User question: {question}
            
            Important guidelines:
            1. Be specific and helpful - don't give generic responses
            2. If recommending repositories, include links and brief descriptions
            3. If discussing issues, include issue numbers and links
            4. If explaining contribution processes, provide step-by-step instructions
            5. Always tailor your response to the user's skill level and interests
            6. Provide actionable next steps the user can take
            7. If you don't have specific information, be honest but still try to be helpful
            """
            
            # Use conversation chain if available, otherwise use direct LLM
            if self.conversation_chain and use_realtime:
                result = self.conversation_chain({"question": enhanced_question})
                return result["answer"]
            else:
                response = self.llm.invoke(enhanced_question).content
                return response
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating response: {error_msg}")
            
            # Provide a graceful fallback response
            return f"""
            I apologize, but I encountered an issue while processing your request. Here's a general answer:
            
            If you're looking to contribute to open source projects, consider:
            
            1. First-time contributor repositories like firstcontributions/first-contributions or up-for-grabs.net
            2. Projects that align with your skills (e.g., Python, JavaScript, documentation)
            3. Look for "good first issue" or "help wanted" labels on GitHub issues
            
            The basic contribution process is:
            - Fork the repository
            - Clone it locally
            - Create a branch for your changes
            - Make and test your changes
            - Submit a pull request
            
            For more specific recommendations, please try rephrasing your question.
            """


class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, OpenSourceChat] = {}
        
    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        chat = OpenSourceChat()
        
        # Expanded curated data with more repositories and contribution advice
        curated_data = [
            "freeCodeCamp/freeCodeCamp: Learn to code for free with millions of learners.",
            "firstcontributions/first-contributions: Helps beginners make their first open-source contribution.",
            "github/docs: GitHub's public documentation - perfect for documentation contributions.",
            "microsoft/vscode: Popular code editor with many well-labeled issues.",
            "rust-lang/rust: Programming language with mentored issues for beginners.",
            "The best way to start contributing is to look for issues labeled 'good first issue' or 'help wanted'.",
            "Don't feel intimidated by large codebases - start with documentation or tests.",
            "Most open source projects are happy to receive contributions from beginners.",
            "Always read the CONTRIBUTING.md file for project-specific guidelines.",
            "When in doubt, ask questions in the issue comments before submitting PR."
        ]
        
        chat.initialize_vectorstore(curated_data)
        self.conversations[conversation_id] = chat
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[OpenSourceChat]:
        return self.conversations.get(conversation_id)
    
    def remove_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

conversation_manager = ConversationManager()

def get_conversation(conversation_id: str) -> OpenSourceChat:
    chat = conversation_manager.get_conversation(conversation_id)
    if not chat:
        raise Exception("Conversation not found")
    return chat

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "version": "1.0"}), 200

@app.route("/start-conversation", methods=["POST"])
def start_conversation():
    try:
        conversation_id = conversation_manager.create_conversation()
        return jsonify({
            "conversation_id": conversation_id,
            "message": "Conversation started successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = chat_request_schema.load(request.get_json())
        conversation_id = data["conversation_id"]
        question = data["question"]
        use_realtime = data["use_realtime"]
        
        # Get conversation
        chat = get_conversation(conversation_id)
        
        # Generate response
        response = chat.get_response(question, use_realtime=use_realtime)
        
        return jsonify({
            "response": response,
            "conversation_id": conversation_id
        }), 200
    except ValidationError as ve:
        print("⚠️ Validation error:", ve.messages)
        return jsonify({"error": ve.messages}), 400
    except Exception as e:
        import traceback
        print("🔥 Server error:", str(e))
        traceback.print_exc()  # SHOW STACKTRACE
        return jsonify({"error": str(e)}), 500

@app.route("/reset-conversation/<conversation_id>", methods=["POST"])
def reset_conversation(conversation_id):
    """Reset a conversation's history but keep the same ID"""
    try:
        # Create a new chat instance
        chat = OpenSourceChat()
        
        # Initialize with curated data
        curated_data = [
            "freeCodeCamp/freeCodeCamp: Learn to code for free with millions of learners.",
            "firstcontributions/first-contributions: Helps beginners make their first open-source contribution.",
            "github/docs: GitHub's public documentation - perfect for documentation contributions.",
            "microsoft/vscode: Popular code editor with many well-labeled issues.",
            "rust-lang/rust: Programming language with mentored issues for beginners."
        ]
        
        chat.initialize_vectorstore(curated_data)
        
        # Replace the existing conversation
        conversation_manager.conversations[conversation_id] = chat
        
        return jsonify({"message": "Conversation reset successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
