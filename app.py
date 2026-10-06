import streamlit as st
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text to Image Generator")
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Text-to-Image Generator")
st.sidebar.info("Model: Stable Diffusion v1.5")
st.sidebar.info("Device: CPU")


st.title("🎨 Text to Image Generator")
st.subheader("Generate images from your imagination using AI")
st.write("Enter a description below and let AI create an image for you.")


prompt = st.text_input(
    "Enter your image prompt:",
    placeholder="Example: A beautiful sunset over snowy mountains"
)
negative_prompt = st.text_input(
    "Negative prompt (optional):",
    placeholder="blurry, distorted, low quality"
)
st.markdown("### 💡 Example Prompts")
st.write("• A futuristic city at night")
st.write("• A cute puppy playing in a garden")
st.write("• A spaceship traveling through space")

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    return pipe.to("cpu")


if st.button("Generate Image", type="primary"):
    if prompt:
        with st.spinner("🤖 AI is creating your image... Please wait."):
            pipe = load_model()

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=15,
                height=512,
                width=512
            ).images[0]

        
        st.subheader("🖼️ Generated Image")
        st.image(image, caption=prompt, use_container_width=True)
        st.download_button(
    label="Download Image",
    data=image.tobytes(),
    file_name="generated_image.png",
    mime="image/png"
)

    else:
        st.warning("Please enter a prompt first.")