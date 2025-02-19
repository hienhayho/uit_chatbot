import chainlit as cl
from pydantic_ai import Agent, Tool

from src.embedding import RAG
from src.settings import Settings
from src.schemas import SupportDependencies, SupportResult


async def response_system(query: str):
    agent = cl.user_session.get("agent")
    res, filenames, results_content = await cl.make_async(agent.contextual_rag_search)(
        query
    )

    print("response_system: ", res)
    return SupportResult(response=res)


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


@cl.on_chat_start
async def start():
    agent = RAG(setting=Settings())
    cl.user_session.set("agent", agent)


@cl.on_message
async def run(message: cl.Message):
    try:
        update_message_history("user", message.content)

        message_history = get_message_history()

        if len(message_history) == 0:
            # print("This is the first message")
            result = support_agent.run_sync(message.content).data
        else:
            # print("This is not the first message")
            result = support_agent.run_sync(
                message.content, message_history=message_history
            ).data

        # result = support_agent.run_sync(message.content, message_history=message_history).data
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

    # agent = cl.user_session.get("agent")
    # res, filenames, results_content = await cl.make_async(agent.contextual_rag_search)(message.content)

    # msg = cl.Message(content="", author="Assistant")
    # response = ""

    # for token in res:
    #     response += token + " "
    #     await msg.stream_token(token)

    # await msg.send()
