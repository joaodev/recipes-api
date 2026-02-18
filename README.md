# Recipe-API

**Recipe-API** is a Django REST Framework application that provides a simple, robust API for managing recipes, ingredients, categories, and user interactions.

## Features

- CRUD operations for recipes, ingredients, and categories
- User authentication and permissions (JWT or Session Auth)
- Filtering, pagination, and search support

## PR Review Automation

This project includes an automated pull request review system powered by AI agents. The system consists of three specialized agents that work together to provide comprehensive code reviews:

### Context Agent
The Context Agent is responsible for gathering all necessary context from the pull request and repository. It collects PR details (author, title, body, diff URL, state, head SHA), changed files, and any additional requested repository files. Once the context is gathered, it hands control to the Commentor Agent.

### Commentor Agent
The Commentor Agent drafts detailed review comments as a human reviewer would. It analyzes the gathered context to write a 200-300 word review in markdown format, covering:
- What is good about the PR
- Compliance with contribution rules
- Availability of tests for new functionality
- Presence of migrations for new models
- Documentation for new endpoints
- Specific lines that could be improved, with quotes and suggestions

The agent directly addresses the author and hands off to the Review and Posting Agent once the draft is complete.

### Review and Posting Agent
The Review and Posting Agent performs a final check on the drafted review to ensure it meets quality criteria (word count, coverage of required sections, quoted lines). If the review is satisfactory, it posts the final comment to the GitHub pull request. If not, it requests rewrites from the Commentor Agent.

To run the PR review agent, use the provided script with appropriate environment variables or command-line arguments for GitHub token, repository, PR number, and OpenAI credentials.

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/recipe-api.git
cd recipe-api

# Install dependencies and activate environment
poetry install
poetry shell
```

## Quickstart

```bash
# Apply database migrations
poetry run python manage.py migrate

# Create a superuser for the admin interface
poetry run python manage.py createsuperuser

# Run the development server
poetry run python manage.py runserver
```

Open your web browser and navigate to:

- `http://localhost:8000/api/recipes` to list all recipes
- `http://localhost:8000/api/recipes` to create a new recipe
- `http://localhost:8000/api/recipes/1` to retrieve a single recipe

## Development

We use Poetry for dependency management and virtual environments.

1. **Install dependencies** (including dev dependencies):
   ```bash
   poetry install
   poetry shell
   ```
2. **Format & Lint**:
   ```bash
   poetry run black . && poetry run isort . && poetry run flake8 .
   ```
3. **Run tests**:
   ```bash
   poetry run pytest
   ```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/the-nulldev/recipes-api/blob/main/CONTRIBUTING.md) for detailed guidelines on issues, pull requests, coding style, and testing.

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/the-nulldev/recipes-api/blob/main/LICENSE) for details.

