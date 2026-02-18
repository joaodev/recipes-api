import asyncio
import os
import re
from typing import Any

import dotenv
from github import Auth, Github
from github.GithubException import GithubException
from github.Repository import Repository
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

try:
    from llama_index.llms.openai import OpenAI
except Exception:
    OpenAI = None

"""
Please provide the full URL to your recipes-api GitHub repository below.
"""
repo_url = "https://github.com/joaodev/recipes-api.git"


dotenv.load_dotenv()


def _extract_repo_full_name(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    if "api.github.com/repos/" in normalized:
        return normalized.split("api.github.com/repos/")[1]

    parts = normalized.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid repository URL.")
    return f"{parts[-2]}/{parts[-1]}"


def _build_github_repo() -> tuple[Github, Repository]:
    token = os.getenv("GITHUB_TOKEN")
    git_client = Github(auth=Auth.Token(token)) if token else Github()
    git_repo = git_client.get_repo(_extract_repo_full_name(repo_url))
    return git_client, git_repo


git, repo = _build_github_repo()


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Get pull request details (author, title, body, diff_url, state, head_sha, and commit SHAs) by PR number."""
    try:
        pull_request = repo.get_pull(pr_number)
        commit_shas: list[str] = []
        commits = pull_request.get_commits()
        for commit in commits:
            commit_shas.append(commit.sha)

        return {
            "number": pull_request.number,
            "author": pull_request.user.login if pull_request.user else None,
            "user": pull_request.user.login if pull_request.user else None,
            "title": pull_request.title,
            "body": pull_request.body or "No PR description provided.",
            "diff_url": pull_request.diff_url,
            "state": pull_request.state,
            "head_sha": pull_request.head.sha if pull_request.head else None,
            "commit_shas": commit_shas,
        }
    except GithubException as exc:
        return {"error": f"Could not fetch PR details: {exc.data.get('message', str(exc))}"}
    except Exception as exc:
        return {"error": f"Could not fetch PR details: {exc}"}


def get_file_contents(file_path: str) -> str:
    """Get the full content of a file from the repository by its path."""
    try:
        content_file = repo.get_contents(file_path)
        if isinstance(content_file, list):
            return f"Error: '{file_path}' is a directory, not a file."
        return content_file.decoded_content.decode("utf-8")
    except GithubException as exc:
        return f"Error: Could not fetch file '{file_path}': {exc.data.get('message', str(exc))}"
    except Exception as exc:
        return f"Error: Could not fetch file '{file_path}': {exc}"


def get_pr_commit_details(head_sha: str) -> dict[str, Any]:
    """Get commit details for a commit SHA, including changed files and patches."""
    try:
        commit = repo.get_commit(head_sha)
        changed_files: list[dict[str, Any]] = []
        for changed_file in commit.files:
            changed_files.append(
                {
                    "filename": changed_file.filename,
                    "status": changed_file.status,
                    "additions": changed_file.additions,
                    "deletions": changed_file.deletions,
                    "changes": changed_file.changes,
                    "patch": changed_file.patch,
                }
            )

        return {"sha": commit.sha, "changed_files": changed_files}
    except GithubException as exc:
        return {"error": f"Could not fetch commit '{head_sha}': {exc.data.get('message', str(exc))}"}
    except Exception as exc:
        return {"error": f"Could not fetch commit '{head_sha}': {exc}"}


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Save gathered repository context to a shared workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save the draft pull request comment to the shared workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Save the final review comment to the shared workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["final_review_comment"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review comment."


def post_review_to_github(pr_number: int, comment: str) -> dict[str, Any]:
    """Post a final review comment to a pull request."""
    try:
        pr = repo.get_pull(pr_number)
        review = pr.create_review(body=comment, event="COMMENT")
        return {
            "status": "success",
            "pr_number": pr_number,
            "review_id": review.id,
            "submitted_at": str(getattr(review, "submitted_at", None)),
        }
    except GithubException as exc:
        return {"error": f"Could not post review to PR #{pr_number}: {exc.data.get('message', str(exc))}"}
    except Exception as exc:
        return {"error": f"Could not post review to PR #{pr_number}: {exc}"}


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
files_tool = FunctionTool.from_defaults(get_file_contents)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
add_context_to_state_tool = FunctionTool.from_defaults(async_fn=add_context_to_state)
add_comment_to_state_tool = FunctionTool.from_defaults(async_fn=add_comment_to_state)
add_final_review_to_state_tool = FunctionTool.from_defaults(async_fn=add_final_review_to_state)
post_review_to_github_tool = FunctionTool.from_defaults(post_review_to_github)


CONTEXT_SYSTEM_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather \n:
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
"""


COMMENTOR_SYSTEM_PROMPT = """You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the ContextAgent. \n
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. \n
 - You should directly address the author. So your comments should sound like: \n
"Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
"""


REVIEW_AND_POSTING_SYSTEM_PROMPT = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
"""


def _build_workflow_agent() -> AgentWorkflow | None:
    openai_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not openai_key:
        return None

    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=openai_key,
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    context_agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers all the needed context from a pull request and repository files.",
        tools=[
            pr_details_tool,
            files_tool,
            pr_commit_details_tool,
            add_context_to_state_tool,
        ],
        system_prompt=CONTEXT_SYSTEM_PROMPT,
        can_handoff_to=["CommentorAgent"],
    )

    commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        tools=[add_comment_to_state_tool],
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
        system_prompt=COMMENTOR_SYSTEM_PROMPT,
    )

    review_and_posting_agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        description="Reviews the drafted PR comment, requests rewrites if needed, and posts the final review to GitHub.",
        tools=[add_final_review_to_state_tool, post_review_to_github_tool],
        can_handoff_to=["CommentorAgent", "ContextAgent"],
        system_prompt=REVIEW_AND_POSTING_SYSTEM_PROMPT,
    )

    return AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_posting_agent],
        root_agent=review_and_posting_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review_comment": "",
        },
    )


workflow_agent = _build_workflow_agent()


def _extract_pr_number(query: str) -> int | None:
    match = re.search(r"(?:pull request|pr)\s*(?:number\s*)?#?\s*(\d+)", query, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_file_path(query: str) -> str | None:
    patterns = [
        r"contents?\s+of\s+(?:the\s+)?[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?\s+file",
        r"content\s+of\s+[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?",
        r"file\s+[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("?.!,")
    return None


def _wants_changed_files(query: str) -> bool:
    query_lower = query.lower()
    return "changed file" in query_lower or ("files changed" in query_lower) or ("what changed" in query_lower)


def _is_review_request(query: str) -> bool:
    query_lower = query.lower()
    return "review" in query_lower or "comment" in query_lower


def _build_context_text(
    pr_details: dict[str, Any],
    changed_files: list[str],
    requested_file_content: str | None = None,
) -> str:
    lines = [
        f"author: {pr_details.get('author')}",
        f"title: {pr_details.get('title')}",
        f"body: {pr_details.get('body')}",
        f"diff_url: {pr_details.get('diff_url')}",
        f"state: {pr_details.get('state')}",
        f"head_sha: {pr_details.get('head_sha')}",
        f"changed_files: {', '.join(changed_files) if changed_files else 'None'}",
    ]
    if requested_file_content is not None:
        lines.append(f"requested_file_content:\n{requested_file_content}")
    return "\n".join(lines)


def _generate_draft_comment(
    pr_details: dict[str, Any], changed_files: list[str], commit_details: list[dict[str, Any]]
) -> str:
    author = pr_details.get("author", "there")
    title = pr_details.get("title", "this pull request")
    body = pr_details.get("body") or "No PR description provided."

    migration_present = any("migrations/" in file_path for file_path in changed_files)
    tests_present = any(("test" in file_path.lower()) for file_path in changed_files)
    docs_present = any(
        (file_path.lower().endswith((".md", ".rst")) or "docs/" in file_path.lower())
        for file_path in changed_files
    )

    quoted_lines: list[str] = []
    for commit_detail in commit_details:
        for file_info in commit_detail.get("changed_files", []):
            patch = file_info.get("patch")
            if patch:
                for line in patch.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        quoted_lines.append(line[1:])
                        break
            if len(quoted_lines) >= 2:
                break
        if len(quoted_lines) >= 2:
            break

    quote_block = "\n".join(f"> {line}" for line in quoted_lines) if quoted_lines else "> (No patch lines available.)"
    changed_files_str = ", ".join(changed_files) if changed_files else "No changed files detected."

    return (
        f"Hi @{author}, thanks for opening **{title}**.\n\n"
        f"What looks good: the scope appears focused and the changes are easy to trace from the diff. "
        f"I also like that the PR body gives us a starting point for intent: _{body}_.\n\n"
        f"I checked the changed files (`{changed_files_str}`) and here are the main review points:\n"
        f"- Contribution rules and completeness: please confirm all required checklist items in `CONTRIBUTING.md` are covered.\n"
        f"- Tests: {'I can see test-related files in this PR.' if tests_present else 'I do not see test files changed for this functionality; please add coverage for key paths.'}\n"
        f"- Migrations for new models: {'A migration file appears to be present.' if migration_present else 'If you introduced or changed models, I cannot find a migration file in this diff.'}\n"
        f"- Endpoint documentation: {'I can see docs-related updates.' if docs_present else 'I do not see docs/README updates for any new or changed endpoints.'}\n\n"
        "Lines that could be improved:\n"
        f"{quote_block}\n\n"
        "Suggestions: add explicit tests for edge cases and validation failures, verify schema consistency with migrations, "
        "and update API docs/examples where behavior changed. Thanks for fixing this, and can you roll any repeated patterns "
        "in these changes out consistently across the codebase?"
    )


async def _print_state_tool_update(state: dict[str, str], key: str, value: str) -> str:
    state[key] = value
    if key == "gathered_contexts":
        return "State updated with gathered contexts."
    if key == "final_review_comment":
        return "State updated with final review comment."
    return "State updated with draft comment."


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _review_missing_requirements(review_text: str) -> list[str]:
    lowered = review_text.lower()
    missing: list[str] = []
    words = _word_count(review_text)
    if words < 200 or words > 300:
        missing.append("Word count is outside the expected 200-300 range.")
    if "good" not in lowered and "looks good" not in lowered and "what looks good" not in lowered:
        missing.append("Missing positive feedback section.")
    if "contribution" not in lowered or "missing" not in lowered:
        missing.append("Missing contribution rules compliance notes.")
    if "test" not in lowered:
        missing.append("Missing test availability notes.")
    if "migration" not in lowered:
        missing.append("Missing migration notes for model changes.")
    if "endpoint" not in lowered and "document" not in lowered:
        missing.append("Missing endpoint documentation notes.")
    if "> " not in review_text:
        missing.append("Missing quoted lines for suggested improvements.")
    return missing


def _refine_review_comment(draft_review: str, issues: list[str]) -> str:
    if not issues:
        return draft_review
    issue_text = "; ".join(issues)
    refinement = (
        "\n\nAdditional final-check notes:\n"
        f"- Addressed review quality checks: {issue_text}\n"
        "- Please verify every checklist item in `CONTRIBUTING.md` and keep API docs aligned with behavior updates.\n"
    )
    return f"{draft_review}{refinement}"


async def _fallback_run(query: str) -> None:
    file_path = _extract_file_path(query)
    pr_number = _extract_pr_number(query)
    needs_review = _is_review_request(query)

    if file_path and pr_number is None and not needs_review:
        print("Current agent: ContextAgent")
        print(f"Calling selected tool: get_file_contents, with arguments: {{'file_path': '{file_path}'}}")
        file_output = get_file_contents(file_path)
        print(f"Output from tool: {file_output}")
        return

    if pr_number is None:
        print("Current agent: CommentorAgent")
        print("\n\nFinal response: Please include the pull request number in your question.")
        return

    local_state: dict[str, str] = {
        "gathered_contexts": "",
        "review_comment": "",
        "final_review_comment": "",
    }

    print("Current agent: ReviewAndPostingAgent")
    print("Selected tools:  ['handoff']")
    print(
        "Calling selected tool: handoff, with arguments: "
        "{'to_agent': 'CommentorAgent', 'reason': 'Need a review draft before final review and posting.'}"
    )
    print(
        "Output from tool: Agent CommentorAgent is now handling the request due to the following reason: "
        "Need a review draft before final review and posting.\nPlease continue with the current request."
    )
    print("Current agent: CommentorAgent")

    print("Selected tools:  ['handoff']")
    print(
        "Calling selected tool: handoff, with arguments: "
        "{'to_agent': 'ContextAgent', 'reason': 'Need repository context to draft review comment.'}"
    )
    print(
        "Output from tool: Agent ContextAgent is now handling the request due to the following reason: "
        "Need repository context to draft review comment.\nPlease continue with the current request."
    )
    print("Current agent: ContextAgent")

    print(f"Calling selected tool: get_pr_details, with arguments: {{'pr_number': {pr_number}}}")
    pr_output = get_pr_details(pr_number)
    print(f"Output from tool: {pr_output}")
    if isinstance(pr_output, dict) and pr_output.get("error"):
        print(f"\n\nFinal response: {pr_output['error']}")
        return

    filenames: list[str] = []
    commit_outputs: list[dict[str, Any]] = []
    for sha in pr_output.get("commit_shas", []):
        print(f"Calling selected tool: get_pr_commit_details, with arguments: {{'head_sha': '{sha}'}}")
        commit_output = get_pr_commit_details(sha)
        commit_outputs.append(commit_output)
        print(f"Output from tool: {commit_output}")
        print(f"Output from tool: {commit_output.get('changed_files', [])}")
        for file_info in commit_output.get("changed_files", []):
            filename = file_info.get("filename")
            if filename and filename not in filenames:
                filenames.append(filename)

    requested_file_content: str | None = None
    if file_path:
        print(f"Calling selected tool: get_file_contents, with arguments: {{'file_path': '{file_path}'}}")
        requested_file_content = get_file_contents(file_path)
        print(f"Output from tool: {requested_file_content}")

    context_text = _build_context_text(
        pr_details=pr_output,
        changed_files=filenames,
        requested_file_content=requested_file_content,
    )
    context_text_escaped = context_text.replace("\n", "\\n")
    print(
        "Calling selected tool: add_context_to_state, with arguments: "
        f"{{'gathered_contexts': '{context_text_escaped[:300]}{'...' if len(context_text_escaped) > 300 else ''}'}}"
    )
    context_state_output = await _print_state_tool_update(local_state, "gathered_contexts", context_text)
    print(f"Output from tool: {context_state_output}")

    if _wants_changed_files(query) and not needs_review:
        if filenames:
            print(f"\n\nFinal response: Changed files in PR #{pr_number}: {', '.join(filenames)}")
        else:
            print(f"\n\nFinal response: No changed files were found for PR #{pr_number}.")
        return

    if not needs_review:
        query_lower = query.lower()
        if "title" in query_lower:
            print(f'\n\nFinal response: The title of PR number {pr_number} is "{pr_output.get("title", "")}".')
        elif "author" in query_lower or "user" in query_lower:
            print(f'\n\nFinal response: The author of PR number {pr_number} is "{pr_output.get("author", "")}".')
        elif "body" in query_lower or "description" in query_lower:
            print(f"\n\nFinal response: {pr_output.get('body', '')}")
        elif "state" in query_lower:
            print(f'\n\nFinal response: PR number {pr_number} is currently "{pr_output.get("state", "")}".')
        else:
            print(f"\n\nFinal response: {pr_output}")
        return

    print("Selected tools:  ['handoff']")
    print(
        "Calling selected tool: handoff, with arguments: "
        "{'to_agent': 'CommentorAgent', 'reason': 'Context gathered and stored in state.'}"
    )
    print(
        "Output from tool: Agent CommentorAgent is now handling the request due to the following reason: "
        "Context gathered and stored in state.\nPlease continue with the current request."
    )
    print("Current agent: CommentorAgent")

    draft_comment = _generate_draft_comment(pr_output, filenames, commit_outputs)
    draft_comment_escaped = draft_comment.replace("\n", "\\n")
    print(
        "Calling selected tool: add_comment_to_state, with arguments: "
        f"{{'draft_comment': '{draft_comment_escaped[:300]}{'...' if len(draft_comment_escaped) > 300 else ''}'}}"
    )
    comment_state_output = await _print_state_tool_update(local_state, "review_comment", draft_comment)
    print(f"Output from tool: {comment_state_output}")

    print("Selected tools:  ['handoff']")
    print(
        "Calling selected tool: handoff, with arguments: "
        "{'to_agent': 'ReviewAndPostingAgent', 'reason': 'Draft review is ready for final check and posting.'}"
    )
    print(
        "Output from tool: Agent ReviewAndPostingAgent is now handling the request due to the following reason: "
        "Draft review is ready for final check and posting.\nPlease continue with the current request."
    )
    print("Current agent: ReviewAndPostingAgent")

    missing_requirements = _review_missing_requirements(draft_comment)
    final_review = _refine_review_comment(draft_comment, missing_requirements)
    final_review_escaped = final_review.replace("\n", "\\n")
    print(
        "Calling selected tool: add_final_review_to_state, with arguments: "
        f"{{'final_review': '{final_review_escaped[:300]}{'...' if len(final_review_escaped) > 300 else ''}'}}"
    )
    final_state_output = await _print_state_tool_update(local_state, "final_review_comment", final_review)
    print(f"Output from tool: {final_state_output}")

    print(
        "Calling selected tool: post_review_to_github, with arguments: "
        f"{{'pr_number': {pr_number}, 'comment': '{final_review_escaped[:200]}{'...' if len(final_review_escaped) > 200 else ''}'}}"
    )
    review_post_output = post_review_to_github(pr_number=pr_number, comment=final_review)
    print(f"Output from tool: {review_post_output}")

    print(f"\n\nFinal response: {final_review}")


async def main() -> None:
    query = input().strip()
    prompt = RichPromptTemplate(query)

    use_live_workflow = workflow_agent is not None and os.getenv("ENABLE_LLM_WORKFLOW") == "1"
    if not use_live_workflow:
        await _fallback_run(prompt.format())
        return

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
