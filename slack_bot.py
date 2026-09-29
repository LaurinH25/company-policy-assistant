"""Slack bot for the Company Policy Assistant.

Runs locally in Socket Mode: the bot opens an outgoing WebSocket to Slack, so no public
URL or hosting is needed. It is online while this script runs.

Ask it by
  - mentioning it in a channel:  @Policy Assistant How many vacation days do I get?
  - the slash command:           /policy How many vacation days do I get?
  - a direct message to the bot

Uses the LLM with a vector index (the preferred approach from the benchmark) and falls back
to rules-based search if Gemini is unavailable.

Run:  python slack_bot.py   (needs SLACK_BOT_TOKEN, SLACK_APP_TOKEN and GEMINI_API_KEY in .env)
"""
import logging
import os
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import policy_core as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("policy-bot")

app = App(token=os.environ["SLACK_BOT_TOKEN"])
POLICY_TEXT = {p.title: p for p in core.POLICIES}
MENTION = re.compile(r"<@[A-Z0-9]+>")


def answer(question: str) -> tuple[core.PolicyResult, bool]:
    """Returns (result, used_fallback)."""
    try:
        return core.llm_vector(question), False
    except Exception:
        log.exception("Gemini failed, falling back to rules-based search")
        return core.rules_based(question), True


def to_blocks(question: str, r: core.PolicyResult, fallback: bool) -> list[dict]:
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*Q:* {question}\n*A:* {r.answer}"}}]
    p = POLICY_TEXT.get(r.policy_title or "")
    if p:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text":
                       f":page_facing_up: *Relevant policy: {p.title}*  _({p.category} · department: {p.department})_\n"
                       f"> {p.policy_text}"}})
    else:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text":
                       ":grey_question: *No policy in the database covers this question.* "
                       "Please contact HR or your manager."}})
    footer = f"{r.approach} · {r.response_time_s:.2f} s · {r.total_tokens:,} tokens"
    if fallback:
        footer = ":warning: Gemini unavailable – keyword-search fallback, check the policy text · " + footer
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]})
    return blocks


def reply(question: str, say, thread_ts: str | None = None) -> None:
    question = MENTION.sub("", question).strip()
    if not question:
        say(text="Ask me a company policy question, e.g. _How many vacation days do I get?_", thread_ts=thread_ts)
        return
    log.info("Question: %s", question)
    r, fallback = answer(question)
    log.info("-> %s (%.2fs, %d tokens)", r.policy_title, r.response_time_s, r.total_tokens)
    say(text=f"{r.answer} (Policy: {r.policy_title or 'none'})",  # plain-text fallback for notifications
        blocks=to_blocks(question, r, fallback), thread_ts=thread_ts)


@app.event("app_mention")
def on_mention(event, say):
    reply(event["text"], say, thread_ts=event.get("thread_ts") or event["ts"])


@app.event("message")
def on_direct_message(event, say):
    # Only plain user messages in DMs; channel messages are handled via @mentions.
    if event.get("channel_type") == "im" and not event.get("subtype") and not event.get("bot_id"):
        reply(event.get("text", ""), say)


@app.command("/policy")
def on_policy_command(ack, command, say):
    ack()  # Slack requires an acknowledgement within 3 seconds
    reply(command.get("text", ""), say)


if __name__ == "__main__":
    core.build_index()
    log.info("Policy bot starting in Socket Mode (Ctrl+C to stop)")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
