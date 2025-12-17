#!/usr/bin/env python3
"""
Comprehensive test to visualize context shrinking and optimization
Shows: token counts, reduction percentages, original vs shrunken context
"""

import json
from context_optimizer import ContextOptimizer
from context_optimizer.tokenizer import estimate_tokens

# Create mock conversation history with lots of code
MOCK_EVENTS = [
    {
        "role": "user",
        "source": "chat",
        "content": "I need to build a user authentication system with JWT tokens. It should have login, logout, and token refresh functionality."
    },
    {
        "role": "assistant",
        "source": "chat",
        "content": "I'll help you build a complete authentication system. Let me start by creating the user model and JWT utilities."
    },
    {
        "role": "tool",
        "source": "file_read",
        "content": """
// src/models/User.ts
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

export interface IUser extends mongoose.Document {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  role: 'admin' | 'user';
  createdAt: Date;
  updatedAt: Date;
  comparePassword(candidatePassword: string): Promise<boolean>;
}

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
  },
  password: {
    type: String,
    required: true,
    minlength: 8,
  },
  firstName: {
    type: String,
    required: true,
  },
  lastName: {
    type: String,
    required: true,
  },
  role: {
    type: String,
    enum: ['admin', 'user'],
    default: 'user',
  },
}, {
  timestamps: true,
});

userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

userSchema.methods.comparePassword = async function(candidatePassword: string): Promise<boolean> {
  return bcrypt.compare(candidatePassword, this.password);
};

export const User = mongoose.model<IUser>('User', userSchema);
"""
    },
    {
        "role": "tool",
        "source": "file_read",
        "content": """
// src/utils/jwt.ts
import jwt from 'jsonwebtoken';
import { IUser } from '../models/User';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-this';
const JWT_EXPIRES_IN = '7d';
const REFRESH_TOKEN_EXPIRES_IN = '30d';

export interface JWTPayload {
  userId: string;
  email: string;
  role: string;
}

export function generateAccessToken(user: IUser): string {
  const payload: JWTPayload = {
    userId: user._id.toString(),
    email: user.email,
    role: user.role,
  };
  
  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN,
  });
}

export function generateRefreshToken(user: IUser): string {
  const payload: JWTPayload = {
    userId: user._id.toString(),
    email: user.email,
    role: user.role,
  };
  
  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: REFRESH_TOKEN_EXPIRES_IN,
  });
}

export function verifyToken(token: string): JWTPayload | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as JWTPayload;
    return decoded;
  } catch (error) {
    console.error('Token verification failed:', error);
    return null;
  }
}

export function decodeToken(token: string): JWTPayload | null {
  try {
    return jwt.decode(token) as JWTPayload;
  } catch (error) {
    return null;
  }
}
"""
    },
    {
        "role": "tool",
        "source": "file_read",
        "content": """
// src/middleware/auth.ts
import { Request, Response, NextFunction } from 'express';
import { verifyToken } from '../utils/jwt';

export interface AuthRequest extends Request {
  user?: {
    userId: string;
    email: string;
    role: string;
  };
}

export function authenticate(req: AuthRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  const token = authHeader.substring(7);
  const payload = verifyToken(token);
  
  if (!payload) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
  
  req.user = payload;
  next();
}

export function requireRole(role: string) {
  return (req: AuthRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    
    if (req.user.role !== role) {
      return res.status(403).json({ error: 'Forbidden: insufficient permissions' });
    }
    
    next();
  };
}
"""
    },
    {
        "role": "tool",
        "source": "file_read",
        "content": """
// src/routes/auth.ts
import express from 'express';
import { User } from '../models/User';
import { generateAccessToken, generateRefreshToken, verifyToken } from '../utils/jwt';
import { authenticate, AuthRequest } from '../middleware/auth';

const router = express.Router();

// Register new user
router.post('/register', async (req, res) => {
  try {
    const { email, password, firstName, lastName } = req.body;
    
    // Validate input
    if (!email || !password || !firstName || !lastName) {
      return res.status(400).json({ error: 'All fields are required' });
    }
    
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ error: 'Email already registered' });
    }
    
    // Create new user
    const user = new User({
      email,
      password,
      firstName,
      lastName,
    });
    
    await user.save();
    
    // Generate tokens
    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);
    
    res.status(201).json({
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
      },
      accessToken,
      refreshToken,
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }
    
    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    // Generate tokens
    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);
    
    res.json({
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
      },
      accessToken,
      refreshToken,
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Refresh token
router.post('/refresh', async (req, res) => {
  try {
    const { refreshToken } = req.body;
    
    if (!refreshToken) {
      return res.status(400).json({ error: 'Refresh token is required' });
    }
    
    const payload = verifyToken(refreshToken);
    if (!payload) {
      return res.status(401).json({ error: 'Invalid or expired refresh token' });
    }
    
    const user = await User.findById(payload.userId);
    if (!user) {
      return res.status(401).json({ error: 'User not found' });
    }
    
    const accessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);
    
    res.json({
      accessToken,
      refreshToken: newRefreshToken,
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get current user profile
router.get('/me', authenticate, async (req: AuthRequest, res) => {
  try {
    const user = await User.findById(req.user?.userId).select('-password');
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ user });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Logout (client-side token removal is primary, this is for server-side tracking)
router.post('/logout', authenticate, async (req: AuthRequest, res) => {
  // In a production app, you might want to blacklist tokens here
  // For now, just return success
  res.json({ message: 'Logged out successfully' });
});

export default router;
"""
    },
    {
        "role": "user",
        "source": "chat",
        "content": "Great! Now I'm getting an error when I try to register a user. Can you help me debug it?"
    },
    {
        "role": "tool",
        "source": "logs",
        "content": """
2024-12-16T10:23:45.123Z [ERROR] Registration failed
2024-12-16T10:23:45.124Z Error: MongoServerError: E11000 duplicate key error collection: mydb.users index: email_1 dup key: { email: "test@example.com" }
    at Connection.onMessage (/node_modules/mongodb/lib/cmap/connection.js:231:30)
    at MessageStream.<anonymous> (/node_modules/mongodb/lib/cmap/connection.js:61:60)
    at MessageStream.emit (node:events:514:28)
    at processIncomingData (/node_modules/mongodb/lib/cmap/message_stream.js:125:16)
    at MessageStream._write (/node_modules/mongodb/lib/cmap/message_stream.js:33:9)
2024-12-16T10:23:45.125Z Request body: {"email":"test@example.com","password":"password123","firstName":"John","lastName":"Doe"}
2024-12-16T10:23:45.126Z Stack trace:
2024-12-16T10:23:45.127Z   at /src/routes/auth.ts:42:15
2024-12-16T10:23:45.128Z   at processTicksAndRejections (node:internal/process/task_queues:95:5)
"""
    },
    {
        "role": "assistant",
        "source": "chat",
        "content": "I see the issue! The error shows that a user with that email already exists in the database. The duplicate key error is being thrown before our existence check. Let me update the registration endpoint to handle this better."
    },
    {
        "role": "tool",
        "source": "file_read",
        "content": """
// src/config/database.ts
import mongoose from 'mongoose';

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/myapp';

export async function connectDatabase() {
  try {
    await mongoose.connect(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('Connected to MongoDB successfully');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
}

export async function disconnectDatabase() {
  try {
    await mongoose.disconnect();
    console.log('Disconnected from MongoDB');
  } catch (error) {
    console.error('MongoDB disconnection error:', error);
  }
}

mongoose.connection.on('error', (error) => {
  console.error('MongoDB connection error:', error);
});

mongoose.connection.on('disconnected', () => {
  console.log('MongoDB disconnected');
});

mongoose.connection.on('connected', () => {
  console.log('MongoDB connected');
});
"""
    },
    {
        "role": "user",
        "source": "chat",
        "content": "Also, I need to add rate limiting to prevent brute force attacks on the login endpoint."
    },
    {
        "role": "tool",
        "source": "file_read",
        "content": """
// src/middleware/rateLimit.ts
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
});

// General API rate limiter
export const apiLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:api:',
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// Stricter limiter for auth endpoints
export const authLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:auth:',
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // limit each IP to 5 requests per windowMs
  message: 'Too many authentication attempts, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true, // don't count successful requests
});

// Very strict limiter for password reset
export const passwordResetLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:reset:',
  }),
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 3, // limit each IP to 3 requests per hour
  message: 'Too many password reset attempts, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});
"""
    }
]


def print_separator(char="=", length=100):
    print(char * length)


def print_section_header(title):
    print_separator()
    print(f"  {title}")
    print_separator()


def print_subsection(title):
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")


def calculate_total_tokens(events):
    """Calculate total tokens in all events"""
    total = 0
    for event in events:
        content = event.get("content", "")
        total += estimate_tokens(content)
    return total


def display_original_context(events):
    """Display all original context with token counts"""
    print_section_header("📚 ORIGINAL CONTEXT (BEFORE OPTIMIZATION)")
    
    total_chars = 0
    total_tokens = 0
    
    for i, event in enumerate(events, 1):
        content = event.get("content", "")
        tokens = estimate_tokens(content)
        chars = len(content)
        total_chars += chars
        total_tokens += tokens
        
        print(f"\n{'─' * 100}")
        print(f"Event #{i} | Role: {event['role']} | Source: {event['source']}")
        print(f"Chars: {chars:,} | Tokens: {tokens:,}")
        print(f"{'─' * 100}")
        print(content[:500] + ("..." if len(content) > 500 else ""))
    
    print(f"\n{'═' * 100}")
    print(f"📊 TOTAL ORIGINAL CONTEXT:")
    print(f"   Total Events: {len(events)}")
    print(f"   Total Characters: {total_chars:,}")
    print(f"   Total Tokens: {total_tokens:,}")
    print(f"{'═' * 100}\n")
    
    return total_tokens


def display_optimized_context(result, query):
    """Display optimized context with detailed metrics"""
    print_section_header("✨ OPTIMIZED CONTEXT (AFTER SHRINKING)")
    
    print(f"\n🔍 Query: '{query}'")
    print(f"\n{'═' * 100}")
    print(f"📊 OPTIMIZATION METRICS:")
    print(f"{'═' * 100}")
    print(f"   Original Tokens:     {result['raw_token_est']:,}")
    print(f"   Optimized Tokens:    {result['optimized_token_est']:,}")
    print(f"   Tokens Saved:        {result['raw_token_est'] - result['optimized_token_est']:,}")
    print(f"   Reduction:           {result['percent_saved_est']:.2f}%")
    print(f"   ───────────────────────────────────────")
    print(f"   Original Chars:      {result['raw_chars']:,}")
    print(f"   Optimized Chars:     {result['optimized_chars']:,}")
    print(f"   Chars Saved:         {result['raw_chars'] - result['optimized_chars']:,}")
    print(f"   ───────────────────────────────────────")
    print(f"   Selected Chunks:     {len(result['optimized_context'])}")
    print(f"{'═' * 100}\n")
    
    for i, item in enumerate(result['optimized_context'], 1):
        reduction_pct = (1 - item['optimized_tokens'] / item['raw_tokens']) * 100 if item['raw_tokens'] > 0 else 0
        
        print(f"\n{'─' * 100}")
        print(f"Chunk #{i} | Type: {item['type']} | Role: {item['role']} | Source: {item['source']}")
        print(f"Original: {item['raw_tokens']:,} tokens ({item['raw_chars']:,} chars)")
        print(f"Shrunk:   {item['optimized_tokens']:,} tokens ({item['optimized_chars']:,} chars)")
        print(f"Saved:    {reduction_pct:.1f}% reduction")
        print(f"{'─' * 100}")
        print(f"OPTIMIZED CONTENT:")
        print(item['summary'])
        print()


def run_visualization_test():
    """Main test function that shows everything"""
    
    print("\n" + "=" * 100)
    print(" " * 30 + "🚀 CONTEXT OPTIMIZATION VISUALIZATION TEST")
    print("=" * 100 + "\n")
    
    # Initialize optimizer
    optimizer = ContextOptimizer()
    session_id = "test-session-123"
    
    # Display original context
    original_tokens = display_original_context(MOCK_EVENTS)
    
    # Ingest events
    print_section_header("📥 INGESTING EVENTS")
    ingest_result = optimizer.ingest(session_id, MOCK_EVENTS)
    print(f"   ✓ Ingested: {ingest_result['ingested']} chunks")
    print(f"   ✓ Deduplicated: {ingest_result['deduped']} duplicates")
    
    # Get stats
    stats = optimizer.get_stats(session_id)
    print(f"   ✓ Total chunks in session: {stats['chunks']}")
    print(f"   ✓ Index terms: {stats['index_terms']}")
    print(f"   ✓ Deduplication rate: {stats['dedup_rate']:.2%}")
    
    # Run optimization with different queries
    queries = [
        "How do I fix the duplicate email error?",
        "Show me the JWT token generation code",
        "How does authentication middleware work?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#' * 100}")
        print(f"  OPTIMIZATION TEST #{i}")
        print(f"{'#' * 100}\n")
        
        result = optimizer.optimize(
            session_id=session_id,
            query_text=query,
            max_chunks=8,
            target_token_budget=2000
        )
        
        display_optimized_context(result, query)
    
    # Final summary
    print_section_header("🎯 FINAL SUMMARY")
    print(f"   Original total tokens across all events:  {original_tokens:,}")
    print(f"   Typical optimized output:                 ~{result['optimized_token_est']:,} tokens")
    print(f"   Average reduction:                        ~{result['percent_saved_est']:.1f}%")
    print(f"   ")
    print(f"   💡 The optimizer intelligently selects relevant chunks based on your query")
    print(f"   💡 Each chunk type (authoritative, diagnostic, etc.) has custom shrinking logic")
    print(f"   💡 Duplicates are removed using SimHash fingerprinting")
    print(f"   💡 Results are cached for performance")
    print_separator()
    
    print("\n✅ Test completed successfully!\n")


if __name__ == "__main__":
    run_visualization_test()
