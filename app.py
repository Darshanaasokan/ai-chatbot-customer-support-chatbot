import streamlit as st
from llama_cpp import Llama

# Load model
@st.cache_resource
def load_model():
    return Llama(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",  # Path to your model
        n_ctx=2048,  # Context window
        n_threads=8,  # Adjust based on CPU cores
        n_batch=512,  # Batch size for speed
        temperature=0.7,  # More focused answers
        top_p=0.9
    )

llm = load_model()

# Streamlit UI
st.set_page_config(page_title="LLaMA 2 Chatbot", layout="centered")
st.title("🤖 AI Chatbot - LLaMA 2 7B")

# Keep chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Format prompt for LLaMA
    system_prompt = "You are a helpful and friendly AI assistant. Keep responses short and relevant."
    chat_history = ""
    for m in st.session_state.messages:
        role = "User" if m["role"] == "user" else "Assistant"
        chat_history += f"{role}: {m['content']}\n"

    formatted_prompt = f"{system_prompt}\n{chat_history}Assistant:"

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            output = llm(
                formatted_prompt,
                max_tokens=256,
                stop=["User:", "Assistant:"]
            )
            response = output["choices"][0]["text"].strip()
            st.markdown(response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
