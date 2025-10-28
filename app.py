from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
import google.generativeai as genai
import uuid
from typing import Dict, Optional, List, Any
import requests
from dotenv import load_dotenv
import base64
import warnings
import re
import json
import time
import datetime

load_dotenv()
warnings.filterwarnings('ignore')

# Attempt to import a lightweight local embeddings backend
try:
	from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore
	_FASTEMBED_AVAILABLE = True
except Exception:
	_FASTEMBED_AVAILABLE = False

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
genai.configure(api_key=GOOGLE_API_KEY)

GITHUB_PER_PAGE = 25

CACHE_CONFIG = {
	"repo": {"expiry": 1800},
	"issue": {"expiry": 900},
	"guide": {"expiry": 3600},
}

REPO_CACHE = {}
ISSUE_CACHE = {}
GUIDE_CACHE = {}

class ChatRequestSchema(Schema):
	conversation_id = fields.Str(required=True)
	question = fields.Str(required=True)
	use_realtime = fields.Bool(missing=True)
	force_refresh = fields.Bool(missing=False)

chat_request_schema = ChatRequestSchema()

class OpenSourceChat:
	def __init__(self):
		self.llm = ChatGoogleGenerativeAI(
			model="gemini-2.5-flash-lite",
			google_api_key=GOOGLE_API_KEY,
			temperature=0.7,
			convert_system_message_to_human=True
		)
		# Prefer FastEmbed (lightweight, no external quota). Fallback to Google embeddings.
		self.embeddings = None
		if _FASTEMBED_AVAILABLE:
			try:
				self.embeddings = FastEmbedEmbeddings()
			except Exception:
				self.embeddings = None
		if self.embeddings is None:
			try:
				self.embeddings = GoogleGenerativeAIEmbeddings(
					model="models/embedding-001",
					google_api_key=GOOGLE_API_KEY
				)
			except Exception:
				self.embeddings = None
		self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,
			chunk_overlap=200,
			length_function=len
		)

		self.message_history = ChatMessageHistory()
		self.memory = ConversationBufferMemory(
			memory_key='chat_history',
			output_key='answer',
			return_messages=True,
			chat_memory=self.message_history
		)

		self.vectorstore = None
		self.conversation_chain = None

		self.user_preferences = {
			"languages": [],
			"interests": [],
			"previous_repos": [],
			"skill_level": "beginner",
			"last_queries": [],
			"preferences_updated": {}
		}

	def initialize_vectorstore(self, curated_data: list[str] = None):
		"""Initialize or update the vector store with curated data. Resilient to failures."""

		# Allow disabling RAG via env for constrained deployments
		if os.environ.get("DISABLE_RAG", "").lower() in {"1", "true", "yes"}:
			self.vectorstore = None
			self.conversation_chain = None
			return

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

		# If embeddings are unavailable (e.g., missing deps or API quota), skip RAG gracefully
		if self.embeddings is None:
			self.vectorstore = None
			self.conversation_chain = None
			return

		chunks = self.text_splitter.split_text("\n".join(all_data))
		try:
			self.vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
			self.conversation_chain = ConversationalRetrievalChain.from_llm(
				llm=self.llm,
				retriever=self.vectorstore.as_retriever(search_kwargs={
					'k': 7,
					'fetch_k': 20,
					'search_type': 'similarity',
				}),
				memory=self.memory,
				return_source_documents=True,
				verbose=True
			)
		except Exception:
			# Any failure building the vectorstore should not take the server down
			self.vectorstore = None
			self.conversation_chain = None

	def add_message_to_history(self, question: str, answer: str):
		"""Add a message pair to the conversation history"""
		self.message_history.add_user_message(question)
		self.message_history.add_ai_message(answer)

	def get_chat_history(self) -> List[dict]:
		"""Get the current chat history in a structured format"""
		messages = self.message_history.messages
		history = []

		for message in messages:
			if isinstance(message, HumanMessage):
				history.append({"role": "human", "content": message.content})
			elif isinstance(message, AIMessage):
				history.append({"role": "ai", "content": message.content})

		return history

	def _extract_language_preferences(self, question: str) -> List[str]:
		"""Extract programming language preferences from user questions with confidence scoring"""
		common_languages = [
			"python", "javascript", "typescript", "java", "c#", "csharp", "c++", "go", "rust",
			"ruby", "php", "kotlin", "swift", "html", "css", "shell", "scala", "r",
			"dart", "lua", "perl", "haskell", "julia", "objective-c", "elixir"
		]

		language_map = {
			"js": "javascript",
			"ts": "typescript",
			"py": "python",
			"csharp": "c#",
			"golang": "go",
			"rb": "ruby",
			"objective-c": "objectivec",
			"c++": "cpp",
			"c plus plus": "cpp"
		}

		found_languages = []

		for lang in common_languages:
			if re.search(r'\b' + re.escape(lang) + r'\b', question.lower()):
				if lang in language_map:
					found_languages.append(language_map[lang])
				else:
					found_languages.append(lang)

		language_patterns = [
			r'(\w+) projects',
			r'(\w+) repositories',
			r'(\w+) developers',
			r'(\w+) programming',
			r'coding in (\w+)',
			r'develop in (\w+)',
			r'(\w+) codebase',
		]

		for pattern in language_patterns:
			matches = re.finditer(pattern, question.lower())
			for match in matches:
				lang = match.group(1)
				if lang in common_languages or lang in language_map:
					final_lang = language_map.get(lang, lang)
					if final_lang not in found_languages:
						found_languages.append(final_lang)

		return found_languages

	def _extract_interests(self, question: str) -> List[str]:
		"""Extract topic interests from user questions with improved pattern matching"""
		common_interests = [
			"web", "mobile", "data science", "machine learning", "ai", "game",
			"database", "frontend", "backend", "fullstack", "devops", "cloud",
			"security", "blockchain", "iot", "embedded", "desktop", "cli",
			"networking", "visualization", "automation", "testing", "docs",
			"documentation", "ui", "ux", "api", "microservices", "serverless",
			"graphics", "audio", "video", "image processing", "nlp"
		]

		interest_phrases = {
			"web development": "web",
			"web dev": "web",
			"website": "web",
			"front end": "frontend",
			"front-end": "frontend",
			"back end": "backend",
			"back-end": "backend",
			"full stack": "fullstack",
			"full-stack": "fullstack",
			"data analysis": "data science",
			"data analytics": "data science",
			"ml": "machine learning",
			"artificial intelligence": "ai",
			"game development": "game",
			"game dev": "game",
			"gaming": "game",
			"devops": "devops",
			"cloud computing": "cloud",
			"cybersecurity": "security",
			"crypto": "blockchain",
			"internet of things": "iot",
			"embedded systems": "embedded",
			"desktop applications": "desktop",
			"desktop app": "desktop",
			"command line": "cli",
			"command-line": "cli",
			"terminal": "cli",
			"network": "networking",
			"data visualization": "visualization",
			"automate": "automation",
			"test": "testing",
			"document": "documentation",
			"docs": "documentation",
			"user interface": "ui",
			"user experience": "ux",
			"apis": "api",
			"rest": "api",
			"graphql": "api",
			"micro services": "microservices",
			"micro-services": "microservices",
			"serverless": "serverless",
			"lambda": "serverless",
			"computer graphics": "graphics",
			"audio processing": "audio",
			"video processing": "video",
			"image": "image processing",
			"natural language processing": "nlp",
			"text processing": "nlp"
		}

		found_interests = []

		for interest in common_interests:
			if interest in question.lower():
				found_interests.append(interest)

		for phrase, canonical in interest_phrases.items():
			if phrase in question.lower() and canonical not in found_interests:
				found_interests.append(canonical)

		return found_interests

	def _extract_skill_level(self, question: str) -> str:
		"""Extract skill level indicators from user questions"""

		current_level = self.user_preferences.get("skill_level", "beginner")

		beginner_phrases = [
			"new to", "beginner", "starting out", "first time", "never contributed",
			"no experience", "newbie", "noob", "just learning", "just started",
			"learning to code", "new developer", "learning programming", "novice"
		]

		intermediate_phrases = [
			"some experience", "intermediate", "familiar with", "worked with",
			"contributed before", "have experience", "comfortable with",
			"have used", "know how to", "proficient"
		]

		advanced_phrases = [
			"advanced", "expert", "very experienced", "senior", "professional",
			"extensive experience", "many years", "maintain", "core contributor",
			"lead developer", "architect"
		]

		for phrase in beginner_phrases:
			if re.search(r'\b' + re.escape(phrase) + r'\b', question.lower()):
				return "beginner"

		for phrase in intermediate_phrases:
			if re.search(r'\b' + re.escape(phrase) + r'\b', question.lower()):
				return "intermediate"

		for phrase in advanced_phrases:
			if re.search(r'\b' + re.escape(phrase) + r'\b', question.lower()):
				return "advanced"

		return current_level

	def _update_user_preferences(self, question: str):
		current_time = time.time()

		self.user_preferences["last_queries"].append(question)
		if len(self.user_preferences["last_queries"]) > 5:
			self.user_preferences["last_queries"].pop(0)

		languages = self._extract_language_preferences(question)
		if languages:
			self.user_preferences["preferences_updated"]["languages"] = current_time
			for lang in languages:
				if lang not in self.user_preferences["languages"]:
					self.user_preferences["languages"].append(lang)
			if len(self.user_preferences["languages"]) > 5:
				self.user_preferences["languages"] = self.user_preferences["languages"][-5:]

		interests = self._extract_interests(question)
		if interests:
			self.user_preferences["preferences_updated"]["interests"] = current_time
			for interest in interests:
				if interest not in self.user_preferences["interests"]:
					self.user_preferences["interests"].append(interest)
			if len(self.user_preferences["interests"]) > 7:
				self.user_preferences["interests"] = self.user_preferences["interests"][-7:]

		skill_level = self._extract_skill_level(question)
		if skill_level != self.user_preferences.get("skill_level", "beginner"):
			self.user_preferences["preferences_updated"]["skill_level"] = current_time
			self.user_preferences["skill_level"] = skill_level

	def search_repositories(self, query: str = "", language: str = "", force_refresh: bool = False) -> list[dict]:
		normalized_query = query.lower().strip() if query else ""
		normalized_language = language.lower().strip() if language else ""
		cache_key = f"{normalized_query}_{normalized_language}"
		current_time = time.time()

		if not force_refresh and cache_key in REPO_CACHE and current_time - REPO_CACHE[cache_key]["timestamp"] < CACHE_CONFIG["repo"]["expiry"]:
			print(f"Using cached repositories for: {cache_key}")
			return REPO_CACHE[cache_key]["data"]

		headers = {
			"Authorization": f"token {GITHUB_TOKEN}",
			"Accept": "application/vnd.github.v3+json"
		}
		url = "https://api.github.com/search/repositories"

		query_parts = []

		if query:
			query_parts.append(query)

		if language:
			query_parts.append(f"language:{language}")
		elif self.user_preferences["languages"] and not language:
			lang_filter = " OR ".join([f"language:{lang}" for lang in self.user_preferences["languages"][:3]])
			query_parts.append(f"({lang_filter})")

		if self.user_preferences["interests"]:
			interests_to_use = self.user_preferences["interests"][:2]
			for interest in interests_to_use:
				query_parts.append(interest)

		skill_level = self.user_preferences.get("skill_level", "beginner")
		if skill_level == "beginner":
			query_parts.append("(good-first-issues:>0 OR help-wanted-issues:>0)")
			query_parts.append("label:\"good first issue\"")

		if skill_level == "beginner":
			query_parts.append("stars:>50")
		elif skill_level == "intermediate":
			query_parts.append("stars:>100")
		else:
			query_parts.append("stars:>500")

		if skill_level == "beginner":
			three_months_ago = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
			query_parts.append(f"pushed:>{three_months_ago}")
		else:
			six_months_ago = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
			query_parts.append(f"pushed:>{six_months_ago}")

		query_parts.append("is:public fork:false archived:false")

		full_query = " ".join(query_parts)

		params = {
			"q": full_query,
			"sort": "stars",
			"order": "desc",
			"per_page": GITHUB_PER_PAGE
		}

		try:
			print(f"Fetching repositories with query: {full_query}")
			response = requests.get(url, headers=headers, params=params, timeout=10)
			response.raise_for_status()

			remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
			if remaining < 10:
				print("WARNING: GitHub API rate limit approaching exhaustion")

			repos_data = response.json()
			if "items" not in repos_data:
				print(f"GitHub API response missing 'items': {repos_data}")
				return []

			repos = repos_data.get("items", [])

			processed_repos = []
			for repo in repos:

				repo_language = repo.get("language") or "Various"

				processed_repo = {
					"name": repo["full_name"],
					"description": repo["description"] or "No description available",
					"url": repo["html_url"],
					"stars": repo["stargazers_count"],
					"forks": repo.get("forks_count", 0),
					"language": repo_language,
					"updated_at": repo["updated_at"],
					"created_at": repo["created_at"],
					"open_issues_count": repo["open_issues_count"],
					"has_issues": repo["has_issues"],
					"topics": repo.get("topics", []),
					"default_branch": repo.get("default_branch", "main")
				}

				processed_repos.append(processed_repo)

			REPO_CACHE[cache_key] = {
				"data": processed_repos,
				"timestamp": current_time,
				"query": full_query
			}

			for repo in processed_repos:
				if repo["name"] not in self.user_preferences["previous_repos"]:
					self.user_preferences["previous_repos"].append(repo["name"])
					if len(self.user_preferences["previous_repos"]) > 15:
						self.user_preferences["previous_repos"].pop(0)

			return processed_repos
		except Exception as e:
			print(f"GitHub API error in search_repositories: {str(e)}")

			return []

	def search_issues(self, repo_full_name: str, force_refresh: bool = False) -> list[dict]:
		"""Search for issues with improved caching and label targeting"""
		cache_key = f"issues_{repo_full_name}"
		current_time = time.time()

		if not force_refresh and cache_key in ISSUE_CACHE and current_time - ISSUE_CACHE[cache_key]["timestamp"] < CACHE_CONFIG["issue"]["expiry"]:
			print(f"Using cached issues for: {repo_full_name}")
			return ISSUE_CACHE[cache_key]["data"]

		headers = {
			"Authorization": f"token {GITHUB_TOKEN}",
			"Accept": "application/vnd.github.v3+json"
		}

		skill_level = self.user_preferences.get("skill_level", "beginner")

		beginner_labels = [
			"good first issue",
			"good-first-issue",
			"beginner",
			"beginner-friendly",
			"easy",
			"help wanted",
			"help-wanted",
			"starter",
			"first-timers-only"
		]

		intermediate_labels = beginner_labels + [
			"enhancement",
			"feature",
			"bug",
			"improvement"
		]

		url = f"https://api.github.com/repos/{repo_full_name}/issues"

		if skill_level == "beginner":

			labels_str = ",".join(beginner_labels)
			params = {
				"labels": labels_str,
				"state": "open",
				"per_page": GITHUB_PER_PAGE
			}
		else:

			labels_str = ",".join(intermediate_labels)
			params = {
				"labels": labels_str,
				"state": "open",
				"per_page": GITHUB_PER_PAGE
			}

		try:
			print(f"Fetching issues for {repo_full_name} with params: {params}")
			response = requests.get(url, headers=headers, params=params, timeout=10)
			response.raise_for_status()

			remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
			if remaining < 10:
				print("WARNING: GitHub API rate limit approaching exhaustion")

			targeted_issues = response.json()

			if len(targeted_issues) < 5:
				print(f"Not enough {skill_level} issues found, fetching regular issues")
				regular_params = {
					"state": "open",
					"per_page": GITHUB_PER_PAGE
				}
				regular_response = requests.get(url, headers=headers, params=regular_params, timeout=10)
				regular_response.raise_for_status()
				regular_issues = regular_response.json()

				targeted_ids = {issue["id"] for issue in targeted_issues}
				combined_issues = targeted_issues + [issue for issue in regular_issues if issue["id"] not in targeted_ids]
			else:
				combined_issues = targeted_issues

			processed_issues = []
			for issue in combined_issues:

				if "pull_request" in issue:
					continue

				last_updated = datetime.datetime.strptime(issue["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
				six_months_ago = datetime.datetime.now() - datetime.timedelta(days=180)
				if last_updated < six_months_ago:
					continue

				labels = []
				for label in issue.get("labels", []):
					if isinstance(label, dict):
						labels.append({
							"name": label.get("name", ""),
							"color": label.get("color", "")
						})
					else:
						labels.append({"name": str(label), "color": ""})

				description = issue.get("body", "")
				if description:

					description = re.sub(r'!\[.*?\]\(.*?\)', '[image]', description)
					description = re.sub(r'```.*?```', '[code block]', description, flags=re.DOTALL)
					description = re.sub(r'\n+', ' ', description)
					description = description[:300] + "..." if len(description) > 300 else description
				else:
					description = "No description available"

				processed_issue = {
					"title": issue["title"],
					"number": issue["number"],
					"url": issue["html_url"],
					"labels": labels,
					"created_at": issue["created_at"],
					"updated_at": issue["updated_at"],
					"comments": issue["comments"],
					"description": description,
					"user": issue["user"]["login"] if "user" in issue else "Unknown",
					"is_beginner_friendly": any(label["name"].lower() in [bl.lower() for bl in beginner_labels]
											  for label in labels)
				}

				processed_issues.append(processed_issue)

			processed_issues.sort(key=lambda x: (not x["is_beginner_friendly"],
												 -datetime.datetime.strptime(x["updated_at"], "%Y-%m-%dT%H:%M:%SZ").timestamp()))

			processed_issues = processed_issues[:15]

			ISSUE_CACHE[cache_key] = {
				"data": processed_issues,
				"timestamp": current_time
			}

			return processed_issues
		except Exception as e:
			print(f"GitHub API error for issues: {str(e)}")
			return []

	def get_contribution_guide(self, repo_full_name: str, force_refresh: bool = False) -> str:
		"""Get contribution guide with improved caching and processing"""
		cache_key = f"guide_{repo_full_name}"
		current_time = time.time()

		if not force_refresh and cache_key in GUIDE_CACHE and current_time - GUIDE_CACHE[cache_key]["timestamp"] < CACHE_CONFIG["guide"]["expiry"]:
			print(f"Using cached contribution guide for: {repo_full_name}")
			return GUIDE_CACHE[cache_key]["data"]

		headers = {
			"Authorization": f"token {GITHUB_TOKEN}",
			"Accept": "application/vnd.github.v3+json"
		}

		guide_paths = [
			"CONTRIBUTING.md",
			".github/CONTRIBUTING.md",
			"docs/CONTRIBUTING.md",
			"CONTRIBUTE.md",
			".github/CONTRIBUTE.md",
			"docs/CONTRIBUTE.md",
			"DEVELOPMENT.md",
			"docs/DEVELOPMENT.md",
			"README.md",
			".github/PULL_REQUEST_TEMPLATE.md"
		]

		guide_content = ""
		found_guide = False

		for path in guide_paths:
			try:
				url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
				response = requests.get(url, headers=headers, timeout=10)

				if response.status_code == 200:
					content_data = response.json()
					if "content" in content_data and content_data["encoding"] == "base64":
						content = base64.b64decode(content_data["content"]).decode("utf-8")

						content = re.sub(r'!\[.*?\]\(.*?\)', '[image]', content)

						if path == "README.md":

							contribution_sections = re.findall(r'(?:##?#?\s+(?:Contribut|Develop|Getting Started|How to|Set up).*?(?=##)|$)(.*?)(?=##|$)',
														   content, re.DOTALL | re.IGNORECASE)
							if contribution_sections:
								content = "\n\n".join(section.strip() for section in contribution_sections if section.strip())
							else:

								content = content[:3000] if len(content) > 3000 else content

						if len(content) > 5000:
							content = content[:5000] + "\n...\n[Guide truncated. See full guide at the repository]"

						file_type = path.split('/')[-1]
						guide_content = f"Contribution guide for {repo_full_name} (from {file_type}):\n\n{content}"
						found_guide = True
						break
			except Exception as e:
				print(f"Error fetching {path} for {repo_full_name}: {str(e)}")
				continue

		if not found_guide:

			try:
				repo_url = f"https://api.github.com/repos/{repo_full_name}"
				repo_response = requests.get(repo_url, headers=headers, timeout=10)
				repo_data = repo_response.json()

				default_branch = repo_data.get("default_branch", "main")
				has_issues = repo_data.get("has_issues", True)
				has_wiki = repo_data.get("has_wiki", False)

				guide_content = f"No specific contribution guide found for {repo_full_name}. Here are general steps to contribute:\n\n"
				guide_content += f"This repository uses '{default_branch}' as its default branch.\n\n"
				guide_content += "1. Fork the repository\n"
				guide_content += "2. Clone your fork locally\n"
				guide_content += f"3. Create a new branch from '{default_branch}' for your feature or bugfix\n"
				guide_content += "4. Make your changes with clear commit messages\n"
				guide_content += "5. Push to your fork\n"
				guide_content += f"6. Submit a pull request to the '{default_branch}' branch of the original repository\n\n"

				if has_issues:
					guide_content += "This repository has Issues enabled. Look for issues labeled 'good first issue' or 'help wanted' for beginner-friendly tasks.\n"
				if has_wiki:
					guide_content += f"This repository has a Wiki which may contain additional documentation: {repo_data.get('html_url')}/wiki\n"

			except Exception as e:
				print(f"Error creating generic guide for {repo_full_name}: {str(e)}")

				guide_content = "No specific contribution guide found. Here are general steps to contribute:\n\n"
				guide_content += "1. Fork the repository\n"
				guide_content += "2. Clone your fork locally\n"
				guide_content += "3. Create a new branch for your feature or bugfix\n"
				guide_content += "4. Make your changes with clear commit messages\n"
				guide_content += "5. Push to your fork\n"
				guide_content += "6. Submit a pull request to the original repository\n\n"
				guide_content += "Look for issues labeled 'good first issue' or 'help wanted' for beginner-friendly tasks."

		GUIDE_CACHE[cache_key] = {
			"data": guide_content,
			"timestamp": current_time
		}

		return guide_content

	def _extract_repo_from_question(self, question: str) -> str:
		"""Extract repository name from question with improved pattern matching"""

		repo_pattern = r'([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)'
		repos = re.findall(repo_pattern, question)

		if repos:
			return repos[0]

		contribute_match = re.search(r'contribute to\s+([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)', question, re.IGNORECASE)
		if contribute_match:
			return contribute_match.group(1)

		issues_match = re.search(r'issues in\s+([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)', question, re.IGNORECASE)
		if issues_match:
			return issues_match.group(1)

		repo_name_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9\-]*/[a-zA-Z0-9\.\-_]+)\s+(?:repo|repository)', question, re.IGNORECASE)
		if repo_name_match:
			return repo_name_match.group(1)

		if self.user_preferences["previous_repos"]:
			return self.user_preferences["previous_repos"][-1]

		return ""

	def crawl_for_open_source_info(self, topic: str = None, language: str = None) -> List[Dict[str, Any]]:
		"""Crawl relevant websites for real-time information about open source projects"""
		import requests
		from bs4 import BeautifulSoup
		import feedparser
		import json
		from urllib.parse import urljoin

		results = []

		if not topic and self.user_preferences["interests"]:
			topic = self.user_preferences["interests"][0]
		if not language and self.user_preferences["languages"]:
			language = self.user_preferences["languages"][0]

		search_query = "open source"
		if topic:
			search_query += f" {topic}"
		if language:
			search_query += f" {language}"

		try:

			github_trending_url = "https://github.com/trending"
			if language:
				github_trending_url += f"/{language}"

			print(f"Crawling GitHub trending: {github_trending_url}")
			response = requests.get(github_trending_url, timeout=10)
			if response.status_code == 200:
				soup = BeautifulSoup(response.text, 'html.parser')
				repo_articles = soup.select('article.Box-row')

				for repo in repo_articles[:5]:
					try:
						repo_link = repo.select_one('h2 a')
						if repo_link:
							href = repo_link.get('href', '').strip('/')
							if href and '/' in href:
								name = href

								desc_elem = repo.select_one('p')
								description = desc_elem.text.strip() if desc_elem else "No description available"

								stars_elem = repo.select_one('a[href$="stargazers"]')
								stars = stars_elem.text.strip() if stars_elem else "N/A"

								results.append({
									"source": "GitHub Trending",
									"name": name,
									"url": f"https://github.com/{name}",
									"description": description,
									"popularity": stars,
									"type": "repository"
								})
					except Exception as e:
						print(f"Error parsing trending repo: {e}")
						continue

			dev_url = f"https://dev.to/search?q={search_query}"
			print(f"Crawling DEV.to: {dev_url}")
			response = requests.get(dev_url, timeout=10)
			if response.status_code == 200:
				soup = BeautifulSoup(response.text, 'html.parser')
				article_cards = soup.select('.crayons-story')

				for article in article_cards[:3]:
					try:
						title_elem = article.select_one('h3.crayons-story__title a')
						if title_elem:
							title = title_elem.text.strip()
							article_url = urljoin("https://dev.to", title_elem.get('href', ''))

							date_elem = article.select_one('time')
							pub_date = date_elem.get('datetime', 'N/A') if date_elem else 'N/A'

							results.append({
								"source": "DEV.to",
								"title": title,
								"url": article_url,
								"published_date": pub_date,
								"type": "article"
							})
					except Exception as e:
						print(f"Error parsing DEV.to article: {e}")
						continue

			rss_feeds = [
				"https://opensource.com/feed",
				"https://changelog.com/feed"
			]

			for feed_url in rss_feeds:
				try:
					print(f"Fetching RSS feed: {feed_url}")
					feed = feedparser.parse(feed_url)
					source = feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else "RSS Feed"

					for entry in feed.entries[:2]:
						results.append({
							"source": source,
							"title": entry.title if hasattr(entry, 'title') else "No title",
							"url": entry.link if hasattr(entry, 'link') else "#",
							"published_date": entry.published if hasattr(entry, 'published') else "N/A",
							"type": "article"
						})
				except Exception as e:
					print(f"Error fetching RSS feed {feed_url}: {e}")
					continue

			if language:
				language_subreddit = language.lower()

				if language_subreddit == "c#":
					language_subreddit = "csharp"
				elif language_subreddit == "c++":
					language_subreddit = "cpp"

				reddit_url = f"https://www.reddit.com/r/{language_subreddit}/top.json?t=week&limit=3"
				headers = {
					"User-Agent": "Mozilla/5.0 OpenSourceGuide/1.0"
				}

				try:
					print(f"Fetching Reddit data: {reddit_url}")
					response = requests.get(reddit_url, headers=headers, timeout=10)
					if response.status_code == 200:
						data = response.json()
						posts = data.get('data', {}).get('children', [])

						for post in posts:
							post_data = post.get('data', {})
							results.append({
								"source": f"Reddit r/{language_subreddit}",
								"title": post_data.get('title', 'No title'),
								"url": f"https://www.reddit.com{post_data.get('permalink', '')}",
								"upvotes": post_data.get('score', 0),
								"type": "discussion"
							})
				except Exception as e:
					print(f"Error fetching Reddit data: {e}")

			try:
				opensource_reddit_url = "https://www.reddit.com/r/opensource/top.json?t=week&limit=3"
				headers = {
					"User-Agent": "Mozilla/5.0 OpenSourceGuide/1.0"
				}

				print(f"Fetching Reddit opensource data")
				response = requests.get(opensource_reddit_url, headers=headers, timeout=10)
				if response.status_code == 200:
					data = response.json()
					posts = data.get('data', {}).get('children', [])

					for post in posts:
						post_data = post.get('data', {})
						results.append({
							"source": "Reddit r/opensource",
							"title": post_data.get('title', 'No title'),
							"url": f"https://www.reddit.com{post_data.get('permalink', '')}",
							"upvotes": post_data.get('score', 0),
							"type": "discussion"
						})
			except Exception as e:
				print(f"Error fetching Reddit opensource data: {e}")

		except Exception as e:
			print(f"Error in web crawling: {str(e)}")

		return results

	def get_project_insights(self, repo_full_name: str, force_refresh: bool = False) -> Dict[str, Any]:
		"""Get deeper insights about a project using GitHub API and web crawling"""
		cache_key = f"insights_{repo_full_name}"
		current_time = time.time()

		if not force_refresh and cache_key in REPO_CACHE and current_time - REPO_CACHE[cache_key]["timestamp"] < CACHE_CONFIG["repo"]["expiry"]:
			print(f"Using cached insights for: {repo_full_name}")
			return REPO_CACHE[cache_key]["data"]

		insights = {
			"repo_name": repo_full_name,
			"contributors": [],
			"commit_frequency": "Unknown",
			"community_profile": {},
			"pull_requests": {
				"open": 0,
				"merged_rate": 0,
				"response_time": "Unknown"
			},
			"technologies": [],
			"related_resources": []
		}

		headers = {
			"Authorization": f"token {GITHUB_TOKEN}",
			"Accept": "application/vnd.github.v3+json"
		}

		try:

			repo_url = f"https://api.github.com/repos/{repo_full_name}"
			repo_response = requests.get(repo_url, headers=headers, timeout=10)
			repo_data = repo_response.json()

			insights["stars"] = repo_data.get("stargazers_count", 0)
			insights["forks"] = repo_data.get("forks_count", 0)
			insights["watchers"] = repo_data.get("subscribers_count", 0) if "subscribers_count" in repo_data else 0
			insights["open_issues"] = repo_data.get("open_issues_count", 0)
			insights["default_branch"] = repo_data.get("default_branch", "main")
			insights["license"] = repo_data.get("license", {}).get("name", "Unknown") if repo_data.get("license") else "Unknown"

			contributors_url = f"https://api.github.com/repos/{repo_full_name}/contributors"
			contributors_response = requests.get(contributors_url, headers=headers, params={"per_page": 5}, timeout=10)
			contributors_data = contributors_response.json()

			if isinstance(contributors_data, list):
				insights["contributors"] = [
					{"login": c.get("login"), "contributions": c.get("contributions"), "url": c.get("html_url")}
					for c in contributors_data
				]

			commits_url = f"https://api.github.com/repos/{repo_full_name}/commits"
			current_date = datetime.datetime.now()
			one_month_ago = (current_date - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

			commits_response = requests.get(
				commits_url,
				headers=headers,
				params={"since": one_month_ago, "per_page": 100},
				timeout=10
			)

			if commits_response.status_code == 200:
				commits_data = commits_response.json()
				if len(commits_data) > 50:
					insights["commit_frequency"] = "Very Active (50+ commits in last month)"
				elif len(commits_data) > 20:
					insights["commit_frequency"] = "Active (20-50 commits in last month)"
				elif len(commits_data) > 5:
					insights["commit_frequency"] = "Moderately Active (5-20 commits in last month)"
				else:
					insights["commit_frequency"] = "Low Activity (< 5 commits in last month)"

			pulls_url = f"https://api.github.com/repos/{repo_full_name}/pulls"
			pulls_response = requests.get(
				pulls_url,
				headers=headers,
				params={"state": "open", "per_page": 100},
				timeout=10
			)

			if pulls_response.status_code == 200:
				open_pulls = pulls_response.json()
				insights["pull_requests"]["open"] = len(open_pulls)

				closed_pulls_response = requests.get(
					pulls_url,
					headers=headers,
					params={"state": "closed", "per_page": 100},
					timeout=10
				)

				if closed_pulls_response.status_code == 200:
					closed_pulls = closed_pulls_response.json()
					merged_count = sum(1 for pr in closed_pulls if pr.get("merged_at") is not None)

					if closed_pulls:
						insights["pull_requests"]["merged_rate"] = round((merged_count / len(closed_pulls)) * 100, 1)

					if closed_pulls:
						response_times = []
						for pr in closed_pulls[:20]:
							created = datetime.datetime.strptime(pr["created_at"], "%Y-%m-%dT%H:%M:%SZ")
							if pr.get("merged_at"):
								closed = datetime.datetime.strptime(pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ")
							else:
								closed = datetime.datetime.strptime(pr["closed_at"], "%Y-%m-%dT%H:%M:%SZ")

							response_time = (closed - created).total_seconds() / 3600
							response_times.append(response_time)

						if response_times:
							avg_time = sum(response_times) / len(response_times)
							if avg_time < 24:
								insights["pull_requests"]["response_time"] = f"Fast (< 24 hours)"
							elif avg_time < 72:
								insights["pull_requests"]["response_time"] = f"Medium (1-3 days)"
							else:
								insights["pull_requests"]["response_time"] = f"Slow (> 3 days)"

			community_url = f"https://api.github.com/repos/{repo_full_name}/community/profile"
			community_headers = {
				"Authorization": f"token {GITHUB_TOKEN}",
				"Accept": "application/vnd.github.black-panther-preview+json"
			}

			community_response = requests.get(community_url, headers=community_headers, timeout=10)
			if community_response.status_code == 200:
				community_data = community_response.json()
				files = community_data.get("files", {})

				community_profile = {
					"has_readme": files.get("readme") is not None,
					"has_contributing": files.get("contributing") is not None,
					"has_code_of_conduct": files.get("code_of_conduct") is not None,
					"has_issue_template": files.get("issue_template") is not None,
					"has_pull_request_template": files.get("pull_request_template") is not None,
					"health_percentage": community_data.get("health_percentage", 0)
				}

				insights["community_profile"] = community_profile

			languages_url = f"https://api.github.com/repos/{repo_full_name}/languages"
			languages_response = requests.get(languages_url, headers=headers, timeout=10)
			if languages_response.status_code == 200:
				languages_data = languages_response.json()

				total_bytes = sum(languages_data.values())
				if total_bytes > 0:
					insights["technologies"] = [
						{"name": lang, "percentage": round((bytes_count / total_bytes) * 100, 1)}
						for lang, bytes_count in sorted(languages_data.items(), key=lambda x: x[1], reverse=True)
					]

			repo_parts = repo_full_name.split('/')
			if len(repo_parts) == 2:
				org_name, repo_name = repo_parts
				related_resources = self.crawl_for_open_source_info(topic=repo_name)
				insights["related_resources"] = related_resources[:5]

			REPO_CACHE[cache_key] = {
				"data": insights,
				"timestamp": current_time
			}

			return insights

		except Exception as e:
			print(f"Error getting project insights: {str(e)}")
			return insights

	def get_stackoverflow_questions(self, repo_name: str = None, topic: str = None) -> List[Dict[str, Any]]:
		"""Get relevant Stack Overflow questions about a repository or topic"""
		if not repo_name and not topic:
			if self.user_preferences["languages"]:
				topic = self.user_preferences["languages"][0]
			elif self.user_preferences["interests"]:
				topic = self.user_preferences["interests"][0]
			else:
				topic = "open source"

		search_term = repo_name if repo_name else topic
		search_term = search_term.replace("/", " ")

		url = "https://api.stackexchange.com/2.3/search"
		params = {
			"order": "desc",
			"sort": "votes",
			"intitle": search_term,
			"site": "stackoverflow",
			"pagesize": 5
		}

		try:
			response = requests.get(url, params=params, timeout=10)
			if response.status_code == 200:
				data = response.json()
				questions = []

				for item in data.get("items", []):
					questions.append({
						"title": item.get("title"),
						"link": item.get("link"),
						"score": item.get("score"),
						"answer_count": item.get("answer_count"),
						"tags": item.get("tags"),
						"is_answered": item.get("is_answered")
					})

				return questions
			else:
				print(f"Stack Overflow API returned status {response.status_code}")
				return []
		except Exception as e:
			print(f"Error fetching Stack Overflow questions: {str(e)}")
			return []

	def get_response(self, question: str, use_realtime: bool = True, force_refresh: bool = False):
		"""Process questions and generate responses with dynamic data and web crawling"""

		self._update_user_preferences(question)

		is_repo_question = any(x in question.lower() for x in ["repository", "repositories", "repos", "projects"])
		is_issue_question = "issue" in question.lower()
		is_contribute_question = any(x in question.lower() for x in ["contribute", "contributing", "contribution"])
		is_guide_question = any(x in question.lower() for x in ["guide", "how to", "steps", "process"])
		is_trend_question = any(x in question.lower() for x in ["trend", "trending", "popular", "new", "latest"])
		is_insight_question = any(x in question.lower() for x in ["insight", "activity", "stats", "statistics", "health"])
		is_help_question = any(x in question.lower() for x in ["help", "assistance", "stuck", "problem", "error"])

		try:

			context_data = {}

			repo_name = self._extract_repo_from_question(question)

			if is_trend_question or "crawl" in question.lower():
				language = None
				topic = None

				if self.user_preferences["languages"]:
					language = self.user_preferences["languages"][0]

				if self.user_preferences["interests"]:
					topic = self.user_preferences["interests"][0]

				extracted_langs = self._extract_language_preferences(question)
				if extracted_langs:
					language = extracted_langs[0]

				extracted_interests = self._extract_interests(question)
				if extracted_interests:
					topic = extracted_interests[0]

				trending_data = self.crawl_for_open_source_info(topic=topic, language=language)
				if trending_data:
					context_data["trending"] = trending_data

					trending_text = []

					sources = {}
					for item in trending_data:
						source = item.get("source", "Unknown")
						if source not in sources:
							sources[source] = []
						sources[source].append(item)

					for source, items in sources.items():
						trending_text.append(f"From {source}:")
						for item in items:
							if item.get("type") == "repository":
								trending_text.append(f"- Repository: [{item.get('name')}]({item.get('url')})")
								if "description" in item:
									trending_text.append(f"  Description: {item.get('description')}")
								if "popularity" in item:
									trending_text.append(f"  Popularity: {item.get('popularity')}")
							else:
								trending_text.append(f"- [{item.get('title')}]({item.get('url')})")
								if "published_date" in item:
									trending_text.append(f"  Published: {item.get('published_date')}")
								if "upvotes" in item:
									trending_text.append(f"  Upvotes: {item.get('upvotes')}")
							trending_text.append("")

					context_data["trending_text"] = "\n".join(trending_text)

			if is_repo_question:

				language = ""
				if self.user_preferences["languages"]:
					language = self.user_preferences["languages"][0]

				query = "good first issue"
				if self.user_preferences["interests"]:
					query += " " + " ".join(self.user_preferences["interests"][:2])

				repos = self.search_repositories(query=query, language=language, force_refresh=force_refresh)
				if repos:
					repo_list = []
					for i, repo in enumerate(repos[:7]):
						repo_list.append(f"- {repo['name']}: {repo['description'][:100]}..." if len(repo['description']) > 100 else f"- {repo['name']}: {repo['description']}")
						repo_list.append(f"  Language: {repo['language']}, Stars: {repo['stars']}, Open Issues: {repo['open_issues_count']}")

					context_data["repositories"] = repos
					context_data["repo_list"] = "\n".join(repo_list)

			if is_issue_question and repo_name:
				issues = self.search_issues(repo_name, force_refresh=force_refresh)
				if issues:
					issue_list = []
					for i, issue in enumerate(issues[:5]):
						label_text = ", ".join([label["name"] for label in issue["labels"][:3]])
						issue_list.append(f"- Issue #{issue['number']}: {issue['title']}")
						issue_list.append(f"  Labels: {label_text if label_text else 'None'}")
						issue_list.append(f"  URL: {issue['url']}")

					context_data["issues"] = issues
					context_data["issue_list"] = "\n".join(issue_list)

			if is_contribute_question or is_guide_question:
				if repo_name:
					guide_content = self.get_contribution_guide(repo_name, force_refresh=force_refresh)
					context_data["contribution_guide"] = guide_content

			if is_insight_question and repo_name:
				insights = self.get_project_insights(repo_name, force_refresh=force_refresh)
				if insights:
					insight_text = [f"Insights for {repo_name}:"]
					insight_text.append(f"- Stars: {insights.get('stars', 'N/A')}")
					insight_text.append(f"- Forks: {insights.get('forks', 'N/A')}")
					insight_text.append(f"- Open Issues: {insights.get('open_issues', 'N/A')}")
					insight_text.append(f"- Activity: {insights.get('commit_frequency', 'Unknown')}")
					insight_text.append(f"- Pull Request Merge Rate: {insights.get('pull_requests', {}).get('merged_rate', 0)}%")
					insight_text.append(f"- PR Response Time: {insights.get('pull_requests', {}).get('response_time', 'Unknown')}")

					community = insights.get('community_profile', {})
					health_score = community.get('health_percentage', 0)
					health_rating = "Excellent" if health_score > 80 else "Good" if health_score > 60 else "Fair" if health_score > 40 else "Poor"
					insight_text.append(f"- Community Health: {health_rating} ({health_score}%)")

					docs = []
					if community.get('has_readme'):
						docs.append("README")
					if community.get('has_contributing'):
						docs.append("CONTRIBUTING")
					if community.get('has_code_of_conduct'):
						docs.append("CODE_OF_CONDUCT")
					doc_status = ", ".join(docs) if docs else "Minimal"
					insight_text.append(f"- Documentation: {doc_status}")

					techs = insights.get('technologies', [])
					if techs:
						tech_list = ", ".join([f"{t['name']} ({t['percentage']}%)" for t in techs[:3]])
						insight_text.append(f"- Top Technologies: {tech_list}")

					context_data["insights"] = insights
					context_data["insight_text"] = "\n".join(insight_text)

			if is_help_question:

				stack_questions = self.get_stackoverflow_questions(repo_name=repo_name)
				if stack_questions:
					question_text = ["Relevant Stack Overflow questions:"]
					for q in stack_questions:
						answered = "✓" if q.get("is_answered") else "✗"
						question_text.append(f"- [{answered}] {q.get('title')}")
						question_text.append(f"  Score: {q.get('score')}, Answers: {q.get('answer_count')}")
						question_text.append(f"  Link: {q.get('link')}")

					context_data["stackoverflow"] = stack_questions
					context_data["stackoverflow_text"] = "\n".join(question_text)

			system_message = """You are GitHelpDesk, an expert assistant specializing in helping users contribute to open source projects.
			Your primary goal is to help users find suitable projects, understand contribution processes, and solve technical issues
			related to open source contribution. Be practical, direct, and provide specific actionable guidance.

			Current User Profile:
			"""

			if self.user_preferences["languages"]:
				system_message += f"\nLanguages: {', '.join(self.user_preferences['languages'])}"
			if self.user_preferences["interests"]:
				system_message += f"\nInterests: {', '.join(self.user_preferences['interests'])}"
			if self.user_preferences["skill_level"]:
				system_message += f"\nSkill Level: {self.user_preferences['skill_level']}"
			if self.user_preferences["previous_repos"]:
				system_message += f"\nPreviously Discussed Repos: {', '.join(self.user_preferences['previous_repos'][-3:])}"

			system_message += "\n\n--- REAL-TIME DATA ---\n"

			if "repositories" in context_data:
				system_message += f"\nREPOSITORIES FOUND:\n{context_data['repo_list']}\n"

			if "issues" in context_data:
				system_message += f"\nISSUES FOUND:\n{context_data['issue_list']}\n"

			if "contribution_guide" in context_data:
				system_message += f"\nCONTRIBUTION GUIDE:\n{context_data['contribution_guide'][:1000]}...\n"

			if "insights" in context_data:
				system_message += f"\nREPOSITORY INSIGHTS:\n{context_data['insight_text']}\n"

			if "stackoverflow_text" in context_data:
				system_message += f"\nRELEVANT QUESTIONS:\n{context_data['stackoverflow_text']}\n"

			if "trending_text" in context_data:
				system_message += f"\nTRENDING DATA:\n{context_data['trending_text']}\n"

			system_message += "\n\nProvide a helpful, informative response based on the real-time data above and your expertise. When recommending repositories or issues, be specific and give actual names and links. If asked about contribution steps, provide detailed guidance tailored to the user's skill level and the specific repository."

			if self.conversation_chain:
				try:

					result = self.conversation_chain({"question": question, "system_message": system_message})
					answer = result["answer"]

					self.add_message_to_history(question, answer)

					return {
						"answer": answer,
						"source_documents": result.get("source_documents", []),
						"context_data": context_data
					}
				except Exception as e:
					print(f"Error using conversation chain: {str(e)}")

					messages = [
						{"role": "system", "content": system_message},
						{"role": "user", "content": question}
					]

					response = self.llm.invoke(messages)
					answer = response.content

					self.add_message_to_history(question, answer)

					return {
						"answer": answer,
						"context_data": context_data
					}
			else:

				if not self.vectorstore:
					self.initialize_vectorstore()

				messages = [
					{"role": "system", "content": system_message},
					{"role": "user", "content": question}
				]

				response = self.llm.invoke(messages)
				answer = response.content

				self.add_message_to_history(question, answer)

				return {
					"answer": answer,
					"context_data": context_data
				}
		except Exception as e:

			print(f"Error in get_response: {str(e)}")
			error_message = "I apologize, but I encountered an error while processing your request. "

			if "rate limit" in str(e).lower():
				error_message += "It seems we've hit GitHub API rate limits. Please try again in a few minutes."
			elif "timeout" in str(e).lower():
				error_message += "There was a timeout while fetching data. Please try again or consider a more specific question."
			else:
				error_message += "Please try a more specific question or check if the repository name is correct."

			return {
				"answer": error_message,
				"error": str(e)
			}

chat_instance = OpenSourceChat()

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
	try:

		data = request.json
		errors = chat_request_schema.validate(data)
		if errors:
			return jsonify({"error": "Invalid request", "details": errors}), 400

		conversation_id = data["conversation_id"]
		question = data["question"]
		use_realtime = data.get("use_realtime", True)
		force_refresh = data.get("force_refresh", False)

		start_time = time.time()
		response = chat_instance.get_response(question, use_realtime, force_refresh)
		end_time = time.time()

		response["processing_time"] = round(end_time - start_time, 2)

		response["conversation_history"] = chat_instance.get_chat_history()

		response["user_preferences"] = chat_instance.user_preferences

		return jsonify(response)
	except ValidationError as ve:
		return jsonify({"error": "Validation error", "details": str(ve)}), 400
	except Exception as e:
		return jsonify({"error": "Server error", "details": str(e)}), 500

@app.route("/api/search/repositories", methods=["GET"])
def search_repositories():
	try:
		query = request.args.get("query", "")
		language = request.args.get("language", "")
		force_refresh = request.args.get("force_refresh", "false").lower() == "true"

		start_time = time.time()
		repos = chat_instance.search_repositories(query, language, force_refresh)
		end_time = time.time()

		return jsonify({
			"repositories": repos,
			"processing_time": round(end_time - start_time, 2)
		})
	except Exception as e:
		return jsonify({"error": "Search error", "details": str(e)}), 500

@app.route("/api/search/issues", methods=["GET"])
def search_issues():
	try:
		repo_name = request.args.get("repo", "")
		force_refresh = request.args.get("force_refresh", "false").lower() == "true"

		if not repo_name:
			return jsonify({"error": "Repository name is required"}), 400

		start_time = time.time()
		issues = chat_instance.search_issues(repo_name, force_refresh)
		end_time = time.time()

		return jsonify({
			"issues": issues,
			"processing_time": round(end_time - start_time, 2)
		})
	except Exception as e:
		return jsonify({"error": "Search error", "details": str(e)}), 500

@app.route("/api/contribution_guide", methods=["GET"])
def get_contribution_guide():
	try:
		repo_name = request.args.get("repo", "")
		force_refresh = request.args.get("force_refresh", "false").lower() == "true"

		if not repo_name:
			return jsonify({"error": "Repository name is required"}), 400

		start_time = time.time()
		guide = chat_instance.get_contribution_guide(repo_name, force_refresh)
		end_time = time.time()

		return jsonify({
			"guide": guide,
			"processing_time": round(end_time - start_time, 2)
		})
	except Exception as e:
		return jsonify({"error": "Error fetching contribution guide", "details": str(e)}), 500

@app.route("/api/project_insights", methods=["GET"])
def get_project_insights():
	try:
		repo_name = request.args.get("repo", "")
		force_refresh = request.args.get("force_refresh", "false").lower() == "true"

		if not repo_name:
			return jsonify({"error": "Repository name is required"}), 400

		start_time = time.time()
		insights = chat_instance.get_project_insights(repo_name, force_refresh)
		end_time = time.time()

		return jsonify({
			"insights": insights,
			"processing_time": round(end_time - start_time, 2)
		})
	except Exception as e:
		return jsonify({"error": "Error fetching project insights", "details": str(e)}), 500

@app.route("/api/trending", methods=["GET"])
def get_trending():
	try:
		topic = request.args.get("topic", None)
		language = request.args.get("language", None)

		start_time = time.time()
		trending_data = chat_instance.crawl_for_open_source_info(topic=topic, language=language)
		end_time = time.time()

		return jsonify({
			"trending": trending_data,
			"processing_time": round(end_time - start_time, 2)
		})
	except Exception as e:
		return jsonify({"error": "Error fetching trending data", "details": str(e)}), 500

@app.route("/api/stackoverflow", methods=["GET"])
def get_stackoverflow():
	try:
		repo_name = request.args.get("repo", None)
		topic = request.args.get("topic", None)

		start_time = time.time()
		questions = chat_instance.get_stackoverflow_questions(repo_name=repo_name, topic=topic)
		end_time = time.time()

		return jsonify({
			"questions": questions,
			"processing_time": round(end_time - start_time, 2)
		})
	except Exception as e:
		return jsonify({"error": "Error fetching Stack Overflow questions", "details": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset_chat():
	"""Reset the chat history and preferences"""
	try:

		global chat_instance
		chat_instance = OpenSourceChat()
		chat_instance.initialize_vectorstore()

		return jsonify({"status": "success", "message": "Chat history and preferences reset"})
	except Exception as e:
		return jsonify({"error": "Error resetting chat", "details": str(e)}), 500

@app.route("/start-conversation", methods=["POST"])
def start_conversation():
	try:

		conversation_id = str(uuid.uuid4())

		return jsonify({
			"status": "success",
			"conversation_id": conversation_id,
			"message": "New conversation started"
		})
	except Exception as e:
		return jsonify({
			"status": "error",
			"message": str(e)
		}), 500

if __name__ == "__main__":

	chat_instance.initialize_vectorstore()

	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
