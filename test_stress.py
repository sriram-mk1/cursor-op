import time
import logging
from context_optimizer import ContextOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    print("="*70)
    print("STRESS TEST: COMPLEX CODE CONTEXT WITH LARGE HISTORY")
    print("="*70)

    # 1. Initialize
    print("\n[1] Initializing ContextOptimizer...")
    start_init = time.time()
    optimizer = ContextOptimizer()
    print(f"    Initialized in {time.time() - start_init:.4f} seconds.")

    # 2. Generate Complex Mock Data
    print("\n[2] Generating Complex Conversation History with Code...")
    session_id = "stress_test_session"
    
    # Simulate a long, complex conversation about building a full-stack app
    conversation = [
        {"role": "user", "content": "I need to build a production-ready FastAPI application with authentication, database, caching, and background jobs."},
        
        {"role": "assistant", "content": """Let's build this step by step. Here's the project structure I recommend:

```
myapp/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       └── users.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py
│   │   └── config.py
│   └── workers/
│       ├── __init__.py
│       └── tasks.py
├── tests/
├── alembic/
├── requirements.txt
└── .env
```

Let's start with the configuration."""},

        {"role": "user", "content": "Show me the config.py implementation."},
        
        {"role": "assistant", "content": """Here's a robust configuration using Pydantic settings:

```python
from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "MyApp API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # Email
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
            )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```"""},

        {"role": "user", "content": "Now show me the database setup with SQLAlchemy 2.0."},
        
        {"role": "assistant", "content": """Here's the modern SQLAlchemy 2.0 setup:

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator
from app.core.config import settings

# Async engine for PostgreSQL
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI.replace("postgresql://", "postgresql+asyncpg://"),
    echo=True,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```"""},

        {"role": "user", "content": "What about the User model with proper relationships?"},
        
        {"role": "assistant", "content": """Here's a comprehensive User model:

```python
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid

# Association table for many-to-many
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', Integer, ForeignKey('roles.id', ondelete='CASCADE'))
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")

class Role(Base):
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    
    users = relationship("User", secondary=user_roles, back_populates="roles")

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="refresh_tokens")
```"""},

        {"role": "user", "content": "I need the authentication system with JWT tokens."},
        
        {"role": "assistant", "content": """Here's a complete JWT authentication implementation:

```python
from datetime import datetime, timedelta
from typing import Optional, Union, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import settings
from app.models.user import User, RefreshToken
from app.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(
    subject: Union[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```"""},

        {"role": "user", "content": "Show me the auth routes with login, register, and refresh token."},
        
        {"role": "assistant", "content": """Here's the complete auth router:

```python
from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.user import User, RefreshToken
from app.schemas.user import UserCreate, UserResponse, Token
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    get_current_user
)
from app.core.config import settings

router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> Any:
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists"
        )
    
    result = await db.execute(select(User).where(User.username == user_in.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="User with this username already exists"
        )
    
    # Create user
    user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user

@router.post("/login", response_model=Token)
async def login(
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    # Authenticate user
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(user.id, expires_delta=access_token_expires)
    
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(user.id, expires_delta=refresh_token_expires)
    
    # Store refresh token
    db_refresh_token = RefreshToken(
        token=refresh_token,
        user_id=user.id,
        expires_at=datetime.utcnow() + refresh_token_expires
    )
    db.add(db_refresh_token)
    await db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    # Verify refresh token
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Check if token exists in DB
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token == refresh_token)
    )
    db_token = result.scalar_one_or_none()
    
    if not db_token or db_token.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token expired or invalid")
    
    # Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(user_id, expires_delta=access_token_expires)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
```"""},

        {"role": "user", "content": "What about Redis caching for frequently accessed data?"},
        
        {"role": "assistant", "content": """Here's a Redis caching implementation:

```python
import json
import redis.asyncio as redis
from typing import Optional, Any
from functools import wraps
from app.core.config import settings

class RedisCache:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        self.redis_client = await redis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
            password=settings.REDIS_PASSWORD,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        
        value = await self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        if not self.redis_client:
            return
        
        await self.redis_client.set(
            key,
            json.dumps(value),
            ex=expire
        )
    
    async def delete(self, key: str):
        if not self.redis_client:
            return
        
        await self.redis_client.delete(key)
    
    async def exists(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        return await self.redis_client.exists(key) > 0

cache = RedisCache()

def cached(expire: int = 3600, key_prefix: str = ""):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, expire)
            
            return result
        
        return wrapper
    return decorator

# Usage example
@cached(expire=300, key_prefix="user")
async def get_user_by_id(user_id: int, db: AsyncSession):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```"""},

        {"role": "user", "content": "Now I need Celery for background tasks like sending emails."},
        
        {"role": "assistant", "content": """Here's a Celery setup with email tasks:

```python
from celery import Celery
from app.core.config import settings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.task_routes = {
    "app.workers.tasks.send_email": "main-queue",
    "app.workers.tasks.process_data": "main-queue",
}

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
)

@celery_app.task(name="send_email", bind=True, max_retries=3)
def send_email(self, to_email: str, subject: str, body: str):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = settings.EMAILS_FROM_EMAIL
        msg['To'] = to_email
        
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_TLS:
                server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        
        return {"status": "sent", "to": to_email}
    
    except Exception as exc:
        self.retry(exc=exc, countdown=60)

@celery_app.task(name="send_verification_email")
def send_verification_email(email: str, token: str):
    verification_url = f"https://myapp.com/verify?token={token}"
    
    html_body = f\"\"\"
    <html>
        <body>
            <h2>Welcome to MyApp!</h2>
            <p>Please verify your email by clicking the link below:</p>
            <a href="{verification_url}">Verify Email</a>
            <p>This link will expire in 24 hours.</p>
        </body>
    </html>
    \"\"\"
    
    send_email.delay(email, "Verify your email", html_body)

@celery_app.task(name="process_large_dataset")
def process_large_dataset(data_id: int):
    # Simulate heavy processing
    import time
    time.sleep(10)
    return {"status": "processed", "data_id": data_id}
```"""},

        {"role": "user", "content": "Can you show me how to add rate limiting to protect the API?"},
        
        {"role": "assistant", "content": """Here's a rate limiting implementation using Redis:

```python
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client, requests_per_minute: int = 60):
        super().__init__(app)
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Get client IP
        client_ip = request.client.host
        
        # Create rate limit key
        current_minute = int(time.time() / 60)
        rate_limit_key = f"rate_limit:{client_ip}:{current_minute}"
        
        # Check current count
        current_count = await self.redis.get(rate_limit_key)
        
        if current_count and int(current_count) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(rate_limit_key)
        pipe.expire(rate_limit_key, 60)
        await pipe.execute()
        
        response = await call_next(request)
        return response

# Decorator for specific endpoints
def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0]
            client_ip = request.client.host
            
            current_window = int(time.time() / window_seconds)
            key = f"rate_limit:{func.__name__}:{client_ip}:{current_window}"
            
            count = await cache.redis_client.get(key)
            
            if count and int(count) >= max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds}s"
                )
            
            pipe = cache.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds)
            await pipe.execute()
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
```"""},

        {"role": "user", "content": "I'm getting errors with the async database connections. How do I debug this?"},
        
        {"role": "assistant", "content": """Here are debugging strategies for async SQLAlchemy:

```python
import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Enable SQLAlchemy logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
logging.getLogger('sqlalchemy.pool').setLevel(logging.DEBUG)

# Add connection pool listeners
@event.listens_for(Engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    logging.info("New DB connection established")

@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    logging.debug("Connection checked out from pool")

@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    logging.debug("Connection returned to pool")

# Common issues and fixes:

# 1. Session not closing properly
async def get_db_safe():
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()  # Always close!

# 2. Accessing lazy-loaded relationships after session close
# Solution: Use eager loading
from sqlalchemy.orm import selectinload

async def get_user_with_posts(user_id: int, db: AsyncSession):
    result = await db.execute(
        select(User)
        .options(selectinload(User.posts))  # Eager load
        .where(User.id == user_id)
    )
    return result.scalar_one_or_none()

# 3. Detached instance errors
# Solution: Refresh or merge
user = await get_user(db)
await db.refresh(user)  # Refresh from DB
# or
user = await db.merge(user)  # Merge detached instance

# 4. Connection pool exhaustion
# Check pool status
async def check_pool_status():
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
    }
```"""},

        {"role": "user", "content": "What about proper error handling and logging throughout the app?"},
        
        {"role": "assistant", "content": """Here's a comprehensive error handling setup:

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from datetime import datetime
import sys

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Custom exception classes
class AppException(Exception):
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class DatabaseException(AppException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)

class AuthenticationException(AppException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

# Exception handlers
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(
        f"AppException: {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    logger.critical(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc()
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Register handlers
def setup_exception_handlers(app: FastAPI):
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
```"""},

        {"role": "user", "content": "Random question: what's the capital of France?"},
        
        {"role": "assistant", "content": "The capital of France is Paris. Now, back to your FastAPI application - is there anything else you'd like to implement?"},
        
        {"role": "user", "content": "Yes, show me how to write comprehensive tests with pytest."},
        
        {"role": "assistant", "content": """Here's a complete pytest setup:

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)

TestSessionLocal = sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@pytest_asyncio.fixture
async def db_session():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "SecurePass123!",
            "full_name": "Test User"
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"
    assert "hashed_password" not in data

@pytest.mark.asyncio
async def test_login_user(client: AsyncClient, db_session: AsyncSession):
    # Create user first
    from app.models.user import User
    from app.core.security import get_password_hash
    
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("password123"),
        is_active=True
    )
    db_session.add(user)
    await db_session.commit()
    
    # Test login
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "test@example.com",
            "password": "password123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_protected_route(client: AsyncClient, db_session: AsyncSession):
    # Create user and get token
    from app.models.user import User
    from app.core.security import get_password_hash, create_access_token
    
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("password123"),
        is_active=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    token = create_access_token(user.id)
    
    # Test protected endpoint
    response = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
```"""},

        {"role": "user", "content": "Perfect! Now I need to understand how to properly configure Celery workers with Redis as the message broker. What are the best practices?"},
    ]
    
    # Add timestamps
    for i, msg in enumerate(conversation):
        msg["ts"] = time.time() + i
        msg["source"] = "chat"

    # Calculate total size
    total_chars = sum(len(msg["content"]) for msg in conversation)
    print(f"    Generated {len(conversation)} messages.")
    print(f"    Total characters: {total_chars:,}")

    # 3. Ingest
    print("\n[3] Ingesting Data...")
    start_ingest = time.time()
    optimizer.ingest(session_id, conversation)
    ingest_time = time.time() - start_ingest
    print(f"    Ingested in {ingest_time:.4f} seconds.")

    # 4. Optimize (Retrieve)
    print("\n[4] Running Optimization...")
    query = "How do I properly configure Celery workers with Redis as the message broker?"
    print(f"    Query: '{query}'")
    
    start_opt = time.time()
    result = optimizer.optimize(
        session_id=session_id,
        query_text=query,
        max_chunks=5,
        target_token_budget=1000
    )
    duration = time.time() - start_opt
    print(f"    Optimization took {duration:.4f} seconds.")

    # 5. Output Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nSTATS:")
    print(f"  - Messages in History: {len(conversation)}")
    print(f"  - Total Characters: {total_chars:,}")
    print(f"  - Raw Token Estimate (Full History): {result['raw_token_est']:,}")
    print(f"  - Optimized Token Count: {result['optimized_token_est']:,}")
    print(f"  - Tokens Reduced: {result['raw_token_est'] - result['optimized_token_est']:,}")
    print(f"  - Percent Saved: {result['percent_saved_est']:.1f}%")
    print(f"  - Ingestion Time: {ingest_time:.4f}s")
    print(f"  - Optimization Time: {duration:.4f}s")
    
    print("\n" + "-"*70)
    print("CONTEXT BEFORE (Full History - First 1000 chars preview)")
    print("-" * 70)
    full_text = "\n".join([f"{m['role']}: {m['content'][:200]}..." for m in conversation[:5]])
    print(full_text)
    print("\n[... " + str(len(conversation) - 5) + " more messages ...]")

    print("\n" + "-"*70)
    print("CONTEXT AFTER (Optimized/Shrunk - Most Relevant Chunks)")
    print("-" * 70)
    for i, chunk in enumerate(result["optimized_context"], 1):
        print(f"\n[Chunk {i}]")
        print(chunk["summary"][:500] + ("..." if len(chunk["summary"]) > 500 else ""))
        print("-" * 70)

    print("\n" + "="*70)
    if duration < 2.0:
        print(f"✅ SUCCESS: Optimization completed in {duration:.4f}s (< 2s target)")
    else:
        print(f"⚠️  WARNING: Optimization took {duration:.2f}s (Target: < 2s)")
    
    if result['percent_saved_est'] > 50:
        print(f"✅ SUCCESS: Achieved {result['percent_saved_est']:.1f}% token reduction")
    else:
        print(f"⚠️  Low reduction: {result['percent_saved_est']:.1f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
