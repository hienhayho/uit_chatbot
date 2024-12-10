import chainlit as cl
from pathlib import Path

from src.settings import Settings
from src.embedding import RAG


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="Hỏi điều luật trong giáo dục", message="Phạm vi điều chính, đối tượng áp dụng")
    ]


@cl.on_chat_start
async def start():
    agent = RAG(setting=Settings())
    
    cl.user_session.set("agent", agent)
    
@cl.on_message
async def run(message: cl.Message):
    agent = cl.user_session.get("agent")
    msg = cl.Message(content="", author="Assistant")
    
    res = await cl.make_async(agent.contextual_rag_search)(message.content)
    
    response = ""
    
    for token in res:
        response += token + " "
        await msg.stream_token(token)
    
    await msg.send()
