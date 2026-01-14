# Dream Decoder

An evidence-informed, AI-powered dream interpretation application with multi-user support and cross-dream pattern analysis.

## Features

### Core Dream Analysis
- **AI-Powered Interpretation**: Uses OpenAI GPT-4o-mini for nuanced, reflective dream analysis
- **Symbol Detection**: Built-in lexicon of 50+ dream symbols with contextual meanings
- **Emotional Mapping**: Tracks emotional states before, during, and after dreams
- **Multi-Layer Analysis**: Micronarrative, summary, and deep interpretive narrative

### Multi-User System
- **User Accounts**: Username/password authentication with secure password hashing
- **Personal Dream Journal**: Each user maintains their own private dream history
- **Session Management**: Secure login sessions with Flask-Login

### Advanced Pattern Detection
- **Dream Threads**: Automatically identifies recurring themes across multiple dreams
- **Cross-Dream Analysis**: Detects narrative arcs, symbol clusters, and emotional patterns
- **Meta-Analysis**: High-level insights about psychological themes and symbolic vocabulary
- **Automatic Triggers**: Thread detection automatically runs every 5 dreams

### User Interface
- **Dream Input**: Comprehensive form capturing dream text, emotions, and life context
- **History Browser**: View all past dreams with timestamps and emotional states
- **Search**: Full-text search across dreams, interpretations, and symbols
- **Thread Viewer**: Explore detected patterns and recurring themes
- **Meta-Analysis Dashboard**: View overarching insights and symbolic trends

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Run the application**:
```bash
python app.py
```

4. **Access in browser**:
```
http://localhost:5000
```

## Architecture

### Backend
- **Flask**: Web framework
- **SQLite**: Database for users, dreams, threads, and meta-analysis
- **OpenAI API**: GPT-4o-mini for dream interpretation and pattern detection
- **Flask-Login**: Session management and authentication

### Database Schema
- **users**: User accounts with hashed passwords
- **dreams**: Individual dream entries with full analysis JSON
- **dream_threads**: Detected patterns across multiple dreams
- **meta_analysis**: High-level insights for each user

### Key Modules
- `app.py`: Main Flask application with routes and authentication
- `database.py`: SQLite operations and data management
- `thread_analyzer.py`: Cross-dream pattern detection and meta-analysis
- `symbol_lexicon.json`: Dream symbol database

## Usage

### First Time Setup
1. Navigate to `/register` to create an account
2. Log in with your username and password
3. Start recording dreams from the home page

### Recording Dreams
1. Enter a dream title (optional)
2. Write your full dream narrative
3. Select emotions during and after the dream
4. Add any relevant life context
5. Click "Decode my dream"

### Viewing Patterns
- **History** (🕒): Browse all your dreams chronologically
- **Search** (🔍): Find dreams by keywords or symbols
- **Threads** (🧵): View recurring patterns (available after 5+ dreams)
- **Meta-Analysis** (📊): See overarching insights (available after 5+ dreams)

## Thread Detection

Threads are automatically detected when you reach dream milestones (5, 10, 15, etc.).

**Thread Types**:
- Recurring situations (e.g., being lost in buildings)
- Emotional arcs (e.g., escalating anxiety patterns)
- Symbol clusters (e.g., water + transformation themes)
- Narrative patterns (e.g., pursuit dreams, threshold-crossing)

## Meta-Analysis

Meta-analysis synthesizes all your dreams to identify:
- Overall psychological themes
- Emotional trajectory over time
- Your unique symbolic vocabulary
- Areas of growth and integration
- Recurring preoccupations

## Security Notes

- Passwords are hashed with SHA-256 + salt
- Each user can only access their own dreams
- Session cookies are used for authentication
- Set a strong SECRET_KEY environment variable in production

## Development

**File Structure**:
```
DreamDecoder/
├── app.py                  # Main application
├── database.py             # Database operations
├── thread_analyzer.py      # Pattern detection
├── requirements.txt        # Dependencies
├── dreams.db              # SQLite database (created on first run)
├── templates/             # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── result.html
│   ├── history.html
│   ├── search.html
│   ├── threads.html
│   └── meta_analysis.html
└── static/                # CSS and JavaScript
    ├── styles.css
    └── app.js
```

## API Costs

This application uses OpenAI's GPT-4o-mini model:
- Dream interpretation: ~1,000-2,000 tokens per dream
- Thread detection: ~2,000-4,000 tokens per analysis
- Meta-analysis: ~3,000-5,000 tokens per generation

Estimate: $0.01-0.02 per dream with current pricing.

## Future Enhancements

Potential features for future development:
- Export dreams to PDF or JSON
- Dream journaling prompts and reminders
- Visualization of emotional trajectories
- Comparison with archetypal dream patterns
- Integration with sleep tracking apps
- Dream sharing and community features (with privacy controls)

## Credits

Built with assistance from Claude (Anthropic) using Claude Code.

## License

This project is for personal use. Please respect OpenAI's usage policies when deploying.
