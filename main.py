
import os
from datetime import datetime, timedelta
from typing import List, Optional
import json
import fitz  # PyMuPDF

import firebase_admin
import google.generativeai as genai
from firebase_admin import credentials, firestore
from fastapi import Depends, FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Firebase Initialization ---
cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "serviceAccountKey.json"))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Gemini AI Initialization ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models ---

# User Models
class UserCreate(BaseModel):
    email: str
    password: str

class UserInDB(BaseModel):
    email: str
    hashed_password: str
    totalPoints: int = 0
    badges: List[str] = []

class User(BaseModel):
    email: str
    totalPoints: int
    badges: List[str] = []

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Quiz Models
class ContentInput(BaseModel):
    text: str
    num_questions: int
    title: str
    creator_id: str
    durationMinutes: Optional[int] = None
    maxAttempts: Optional[int] = None
    isAdaptive: bool = False

class Question(BaseModel):
    question_number: int
    question_text: str
    options: Optional[List[str]] = None
    answer: str
    question_type: str  # "multiple_choice" or "short_answer"

class Quiz(BaseModel):
    id: Optional[str] = None
    title: str
    creator_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    questions: List[Question]
    durationMinutes: Optional[int] = None
    maxAttempts: Optional[int] = None
    source_text: Optional[str] = None
    isAdaptive: bool = False

# Submission Models
class UserAnswer(BaseModel):
    question_number: int
    answer: str
    confidence: Optional[str] = None

class EvaluationPayload(BaseModel):
    source_text: str
    questions: List[Question]
    user_answers: List[UserAnswer]

class Feedback(BaseModel):
    question_number: int
    user_answer: str
    ai_feedback: str
    correct: bool
    confidence: Optional[str] = None

class Result(BaseModel):
    score: float
    summary: str
    feedback: List[Feedback]

class QuizSubmission(BaseModel):
    id: Optional[str] = None
    quiz_id: str
    user_id: str
    quiz_title: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    results: Result
    user_answers: List[UserAnswer]
    attemptNumber: Optional[int] = None

class QuizSubmissionResponse(BaseModel):
    submission: QuizSubmission
    points_earned: int
    new_badges_unlocked: List[str]

class RecommendationsResponse(BaseModel):
    recommendations: List[str]

# Adaptive Quiz Models
class AdaptiveAnswerPayload(BaseModel):
    session_id: str
    question: Question
    answer: str
    current_difficulty: str
    confidence: Optional[str] = None

class AdaptiveAnswerResponse(BaseModel):
    is_correct: bool
    explanation: str
    next_question: Optional[Question] = None

class StartAdaptiveResponse(BaseModel):
    question: Question
    session_id: str


# --- Authentication Utilities ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user_ref = db.collection('users').document(token_data.email).get()
    if user_ref.exists:
        user_data = user_ref.to_dict()
        user_data.setdefault('totalPoints', 0)
        user_data.setdefault('badges', [])
        return User(**user_data)
    else:
        raise credentials_exception


# --- API Endpoints ---

# Authentication Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_ref = db.collection('users').document(form_data.username).get()
    if not user_ref.exists:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = user_ref.to_dict()
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/")
async def create_user(user: UserCreate):
    user_ref = db.collection('users').document(user.email).get()
    if user_ref.exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    hashed_password = get_password_hash(user.password)
    user_in_db = UserInDB(email=user.email, hashed_password=hashed_password, totalPoints=0, badges=[])
    db.collection('users').document(user.email).set(user_in_db.dict())
    return {"email": user.email}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Quiz Generation Endpoint
@app.post("/generate-quiz", response_model=Quiz)
async def generate_quiz(content: ContentInput):
    if content.isAdaptive:
        quiz = Quiz(
            title=content.title,
            creator_id=content.creator_id,
            questions=[],
            durationMinutes=content.durationMinutes,
            maxAttempts=content.maxAttempts,
            source_text=content.text,
            isAdaptive=True
        )
        quiz_ref = db.collection('quizzes').document()
        quiz.id = quiz_ref.id
        quiz_ref.set(quiz.dict())
        return quiz

    prompt = f"""
    Based on the following text, generate a quiz with {content.num_questions} questions.
    For each question, provide the following fields in a JSON object:
    - "question_number": an integer representing the question number.
    - "question_text": a string containing the question.
    - "options": a list of strings for multiple choice questions, or null for short answer questions.
    - "answer": a string containing the correct answer.
    - "question_type": a string, either "multiple_choice" or "short_answer".

    Format the output as a single JSON array of these objects.

    Example for a single question:
    {{
        "question_number": 1,
        "question_text": "What is the capital of France?",
        "options": ["London", "Paris", "Berlin", "Madrid"],
        "answer": "Paris",
        "question_type": "multiple_choice"
    }}

    Text:
    {content.text}
    """
    try:
        response = model.generate_content(prompt)
        questions_data = response.text.strip()
        if questions_data.startswith("```json"):
            questions_data = questions_data[7:-3]
        
        questions = [Question(**q) for q in json.loads(questions_data)]
        
        quiz = Quiz(
            title=content.title,
            creator_id=current_user.email,
            questions=questions,
            durationMinutes=content.durationMinutes,
            maxAttempts=content.maxAttempts,
            source_text=content.text
        )
        
        # Save to Firestore
        quiz_ref = db.collection('quizzes').document()
        quiz.id = quiz_ref.id
        quiz_ref.set(quiz.dict())
        
        return quiz

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate or save quiz: {e}")

@app.post("/generate-quiz-from-file", response_model=Quiz)
async def generate_quiz_from_file(
    file: UploadFile = File(...),
    title: str = Form(...),
    num_questions: int = Form(...),
    creator_id: str = Form(...),
    durationMinutes: Optional[int] = Form(None),
    maxAttempts: Optional[int] = Form(None),
):
    try:
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        prompt = f"""
        Based on the following text, generate a quiz with {num_questions} questions.
        For each question, provide the following fields in a JSON object:
        - "question_number": an integer representing the question number.
        - "question_text": a string containing the question.
        - "options": a list of strings for multiple choice questions, or null for short answer questions.
        - "answer": a string containing the correct answer.
        - "question_type": a string, either "multiple_choice" or "short_answer".

        Format the output as a single JSON array of these objects.

        Example for a single question:
        {{
            "question_number": 1,
            "question_text": "What is the capital of France?",
            "options": ["London", "Paris", "Berlin", "Madrid"],
            "answer": "Paris",
            "question_type": "multiple_choice"
        }}

        Text:
        {text}
        """
        response = model.generate_content(prompt)
        questions_data = response.text.strip()
        if questions_data.startswith("```json"):
            questions_data = questions_data[7:-3]
        
        questions = [Question(**q) for q in json.loads(questions_data)]
        
        quiz = Quiz(
            title=title,
            creator_id=creator_id,
            questions=questions,
            durationMinutes=durationMinutes,
            maxAttempts=maxAttempts,
            source_text=text
        )
        
        # Save to Firestore
        quiz_ref = db.collection('quizzes').document()
        quiz.id = quiz_ref.id
        quiz_ref.set(quiz.dict())
        
        return quiz

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate or save quiz from file: {e}")


# Adaptive Quiz Endpoints
@app.post("/quizzes/{quiz_id}/start_adaptive", response_model=StartAdaptiveResponse)
async def start_adaptive_quiz(quiz_id: str, current_user: User = Depends(get_current_user)):
    quiz_ref = db.collection('quizzes').document(quiz_id).get()
    if not quiz_ref.exists:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = Quiz(**quiz_ref.to_dict())
    if not quiz.isAdaptive:
        raise HTTPException(status_code=400, detail="This is not an adaptive quiz.")

    # Generate first question (medium difficulty)
    prompt = f"""
    Based on the following text, generate a single quiz question with a medium difficulty.
    Text: {quiz.source_text}
    Format the output as a single JSON object with the following fields:
    - "question_number": 1
    - "question_text": string
    - "options": list of strings or null
    - "answer": string
    - "question_type": "multiple_choice" or "short_answer"
    """
    try:
        response = model.generate_content(prompt)
        question_data = response.text.strip()
        if question_data.startswith("```json"):
            question_data = question_data[7:-3]
        
        question = Question(**json.loads(question_data))

        # Create adaptive session
        session_ref = db.collection('adaptive_sessions').document()
        session_id = session_ref.id
        session_ref.set({
            "quiz_id": quiz_id,
            "user_id": current_user.email,
            "score": 0,
            "questions_answered": [],
            "last_difficulty": "medium"
        })

        return StartAdaptiveResponse(question=question, session_id=session_id)

    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=500, detail=f"Failed to start adaptive quiz: {e}")

@app.post("/quizzes/submit_adaptive_answer", response_model=AdaptiveAnswerResponse)
async def submit_adaptive_answer(payload: AdaptiveAnswerPayload, current_user: User = Depends(get_current_user)):
    session_ref = db.collection('adaptive_sessions').document(payload.session_id)
    session = session_ref.get()
    if not session.exists:
        raise HTTPException(status_code=404, detail="Adaptive quiz session not found.")

    # Basic answer evaluation (can be improved with AI)
    is_correct = payload.answer.lower() == payload.question.answer.lower()
    explanation = "Correct!" if is_correct else f"The correct answer is: {payload.question.answer}"

    # Determine next difficulty
    if is_correct:
        if payload.current_difficulty == "easy":
            next_difficulty = "medium"
        else: # medium or hard
            next_difficulty = "hard"
    else: # incorrect
        if payload.current_difficulty == "hard":
            next_difficulty = "medium"
        else: # medium or easy
            next_difficulty = "easy"

    # Update session
    answered_question = payload.question.dict()
    answered_question['user_answer'] = payload.answer
    answered_question['confidence'] = payload.confidence
    answered_question['is_correct'] = is_correct

    session_ref.update({
        "score": firestore.Increment(1) if is_correct else firestore.Increment(0),
        "questions_answered": firestore.ArrayUnion([answered_question]),
        "last_difficulty": next_difficulty
    })

    # Generate next question
    quiz_ref = db.collection('quizzes').document(session.to_dict()['quiz_id']).get()
    quiz = Quiz(**quiz_ref.to_dict())

    prompt = f"""
    Based on the following text, generate a single quiz question with a {next_difficulty} difficulty.
    Text: {quiz.source_text}
    Format the output as a single JSON object with the following fields:
    - "question_number": {len(session.to_dict()['questions_answered']) + 1}
    - "question_text": string
    - "options": list of strings or null
    - "answer": string
    - "question_type": "multiple_choice" or "short_answer"
    """
    try:
        response = model.generate_content(prompt)
        question_data = response.text.strip()
        if question_data.startswith("```json"):
            question_data = question_data[7:-3]
        
        next_question = Question(**json.loads(question_data))

        return AdaptiveAnswerResponse(
            is_correct=is_correct,
            explanation=explanation,
            next_question=next_question
        )

    except (json.JSONDecodeError, Exception) as e:
        # Could not generate next question, end of quiz
        return AdaptiveAnswerResponse(
            is_correct=is_correct,
            explanation=explanation,
            next_question=None
        )


# Quiz Management Endpoints
@app.get("/quizzes/", response_model=List[Quiz])
async def get_quizzes(current_user: User = Depends(get_current_user)):
    quizzes_ref = db.collection('quizzes').where('creator_id', '==', current_user.email).stream()
    quizzes = []
    for quiz in quizzes_ref:
        q_data = quiz.to_dict()
        q_data['id'] = quiz.id
        quizzes.append(Quiz(**q_data))
    return quizzes

@app.get("/quizzes/{quiz_id}", response_model=Quiz)
async def get_quiz(quiz_id: str, current_user: User = Depends(get_current_user)):
    quiz_ref = db.collection('quizzes').document(quiz_id).get()
    if not quiz_ref.exists:
        raise HTTPException(status_code=404, detail="Quiz not found")
    quiz = quiz_ref.to_dict()
    quiz['id'] = quiz_ref.id
    return Quiz(**quiz)

# Quiz Evaluation Endpoint
@app.post("/quizzes/{quiz_id}/submit", response_model=QuizSubmissionResponse)
async def submit_quiz(quiz_id: str, user_answers: List[UserAnswer], current_user: User = Depends(get_current_user)):
    quiz_ref = db.collection('quizzes').document(quiz_id).get()
    if not quiz_ref.exists:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = Quiz(**quiz_ref.to_dict())

    if quiz.maxAttempts and quiz.maxAttempts > 0:
        submissions_ref = db.collection('submissions').where('user_id', '==', current_user.email).where('quiz_id', '==', quiz_id).stream()
        attempt_count = len(list(submissions_ref))
        if attempt_count >= quiz.maxAttempts:
            raise HTTPException(status_code=403, detail="You have reached the maximum number of attempts for this quiz.")
        attempt_number = attempt_count + 1
    else:
        attempt_number = 1


    prompt = f"""
    Source Text:
    {quiz.title} 

    Questions:
    {[q.dict() for q in quiz.questions]}

    User Answers:
    {[ua.dict() for ua in user_answers]}

    Please evaluate the user's answers. For each question, provide the following:
    - question_number: The number of the question.
    - user_answer: The answer provided by the user.
    - confidence: The user's confidence level for the answer.
    - ai_feedback: A brief explanation of why the answer is correct or incorrect.
    - correct: A boolean indicating if the user's answer is correct.
    
    Then, provide an overall score (0-100) and a summary of learning recommendations.
    Format the output as a JSON object with 'score', 'summary', and 'feedback' fields.
    The 'feedback' field should be a list of objects with the structure defined above.
    """
    
    response = model.generate_content(prompt)
    
    try:
        evaluation_data = response.text.strip()
        if evaluation_data.startswith("```json"):
            evaluation_data = evaluation_data[7:-3]
            
        result_data = json.loads(evaluation_data)
        result = Result(**result_data)
        
        submission = QuizSubmission(
            quiz_id=quiz_id,
            user_id=current_user.email,
            quiz_title=quiz.title,
            results=result,
            user_answers=user_answers,
            attemptNumber=attempt_number
        )
        
        # Save to Firestore
        submission_ref = db.collection('submissions').document()
        submission.id = submission_ref.id
        submission_ref.set(submission.dict())

        # Gamification logic
        points_earned = sum(10 for item in result.feedback if item.correct)
        new_badges_unlocked = []

        user_ref = db.collection('users').document(current_user.email)
        user_ref.update({"totalPoints": firestore.Increment(points_earned)})

        if result.score == 100 and "Perfect Score" not in current_user.badges:
            new_badges_unlocked.append("Perfect Score")
            user_ref.update({"badges": firestore.ArrayUnion(["Perfect Score"])})

        user_submissions = db.collection('submissions').where('user_id', '==', current_user.email).get()
        if len(user_submissions) == 1 and "First Quiz" not in current_user.badges:
            new_badges_unlocked.append("First Quiz")
            user_ref.update({"badges": firestore.ArrayUnion(["First Quiz"])})
        
        return QuizSubmissionResponse(
            submission=submission,
            points_earned=points_earned,
            new_badges_unlocked=new_badges_unlocked
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate or save submission: {e}")


# Recommendations Endpoint
@app.get("/recommendations/{submission_id}", response_model=RecommendationsResponse)
async def get_recommendations(submission_id: str, current_user: User = Depends(get_current_user)):
    # 1. Fetch submission
    submission_ref = db.collection('submissions').document(submission_id).get()
    if not submission_ref.exists:
        raise HTTPException(status_code=404, detail="Submission not found")
    submission = submission_ref.to_dict()

    # 2. Authorize user
    if submission['user_id'] != current_user.email:
        raise HTTPException(status_code=403, detail="Not authorized to view recommendations for this submission")

    # 3. Fetch quiz to get source text
    quiz_id = submission['quiz_id']
    quiz_ref = db.collection('quizzes').document(quiz_id).get()
    if not quiz_ref.exists:
        raise HTTPException(status_code=404, detail="Quiz not found")
    quiz = quiz_ref.to_dict()
    source_text = quiz.get('source_text')

    if not source_text:
        raise HTTPException(status_code=404, detail="Source text for this quiz not found.")

    # 4. Identify incorrect answers
    incorrect_answers = [
        feedback for feedback in submission['results']['feedback'] if not feedback['correct']
    ]

    if not incorrect_answers:
        return RecommendationsResponse(recommendations=["Great job! You answered all questions correctly. No specific recommendations at this time."])

    # 5. Construct prompt for Gemini
    prompt = f"""
    As an expert tutor, please provide personalized learning recommendations for a user based on their incorrect quiz answers.

    Original Source Text:
    ---
    {source_text}
    ---

    The user was quizzed on the above text and answered the following questions incorrectly:
    """

    for item in incorrect_answers:
        question_number = item['question_number']
        # Find the question text from the quiz data
        question_text = next((q['question_text'] for q in quiz['questions'] if q['question_number'] == question_number), "Question not found")
        user_answer = item['user_answer']
        prompt += f"\n- Question: \"{question_text}\"\n  User's Incorrect Answer: \"{user_answer}\"\n"

    prompt += """
    Based on these specific mistakes, generate 3-5 actionable and specific learning recommendations.
    Focus on the underlying concepts the user seems to be struggling with.
    Frame the recommendations as helpful tips for what to review in the source text or what concepts to focus on.

    Format the output as a JSON object with a single key "recommendations" which contains a list of strings.
    Example:
    {
        "recommendations": [
            "Review the section on 'X' to better understand the difference between A and B.",
            "It seems you're confusing concept 'Y' with 'Z'. Pay close attention to the definitions in the first part of the text.",
            "Practice identifying the key characteristics of 'W' as described in the source material."
        ]
    }
    """

    # 6. Send to Gemini
    try:
        response = model.generate_content(prompt)
        recommendations_data = response.text.strip()
        if recommendations_data.startswith("```json"):
            recommendations_data = recommendations_data[7:-3]

        recommendations_json = json.loads(recommendations_data)
        return RecommendationsResponse(**recommendations_json)

    except (json.JSONDecodeError, Exception) as e:
        # Fallback response if Gemini fails or returns malformed data
        return RecommendationsResponse(recommendations=["Could not generate recommendations at this time. Please try again later."])


# Gamification Endpoints
@app.get("/leaderboard", response_model=List[User])
async def get_leaderboard(current_user: User = Depends(get_current_user)):
    users_ref = db.collection('users').order_by('totalPoints', direction=firestore.Query.DESCENDING).limit(20).stream()
    leaderboard = [User(**user.to_dict()) for user in users_ref]
    return leaderboard


# Results Endpoints
@app.get("/submissions/", response_model=List[QuizSubmission])
async def get_submissions(current_user: User = Depends(get_current_user)):
    submissions_ref = db.collection('submissions').where('user_id', '==', current_user.email).stream()
    submissions = []
    for sub in submissions_ref:
        s_data = sub.to_dict()
        s_data['id'] = sub.id
        submissions.append(QuizSubmission(**s_data))
    return submissions

@app.get("/submissions/{submission_id}", response_model=QuizSubmission)
async def get_submission(submission_id: str, current_user: User = Depends(get_current_user)):
    submission_ref = db.collection('submissions').document(submission_id).get()
    if not submission_ref.exists:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    submission = submission_ref.to_dict()
    submission['id'] = submission_ref.id
    
    # Ensure the submission belongs to the current user
    if submission['user_id'] != current_user.email:
        raise HTTPException(status_code=403, detail="Not authorized to view this submission")
        
    return QuizSubmission(**submission)

