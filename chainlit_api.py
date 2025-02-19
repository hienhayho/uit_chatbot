import os
import requests
import chainlit as cl
from pydantic_ai import Agent, Tool

from schemas import SupportDependencies, SupportResult

url_api = os.getenv("SYSTEM_API")


async def response_system(query: str):
    res = requests.post(url_api, json={"content": query})
    response = res.json()["result"]
    print("response_system: ", response)
    return SupportResult(response=response)


support_agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        "You are a supportive university admissions counselor chatbot. Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, etc.—while providing clear, empathetic guidance and assessing query urgency."
    ),
    tools=[
        Tool(response_system),
    ],
)


def get_message_history():
    """Retrieve message history from user session."""

    message_history = cl.user_session.get("message_history")

    if not isinstance(message_history, list):  # Ensure it's always a list
        message_history = []
        cl.user_session.set("message_history", message_history)

    return message_history


def update_message_history(role: str, content: str):
    """Update message history and store it in user session."""
    message_history = get_message_history()
    message_history.extend(support_agent.run_sync(content).all_messages())
    cl.user_session.set("message_history", message_history)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Tuyển sinh ở UIT cần lưu ý những gì",
            message="Tuyển sinh ở UIT cần lưu ý những gì",
        )
    ]


# @cl.on_chat_start
# async def start():
#     agent = RAG(setting=Settings())
#     cl.user_session.set("agent", agent)


@cl.on_message
async def run(message: cl.Message):
    try:
        print("message: ", message.content)
        update_message_history("user", message.content)

        message_history = get_message_history()

        if len(message_history) == 0:
            result = support_agent.run_sync(message.content).data
        else:
            result = support_agent.run_sync(
                message.content, message_history=message_history
            ).data

            print("result: ", result)

        query = result.response

        msg = cl.Message(content="", author="Assistant")
        response = ""

        for token in query:
            response += token + " "
            await msg.stream_token(token)

        await msg.send()

        update_message_history("assistant", response)

    except AssertionError as e:
        print("ERROR: AssertionError encountered:", e)
    except Exception as e:
        print("ERROR: Unexpected error:", e)
