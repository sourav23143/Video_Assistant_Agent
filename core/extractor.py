#Actionableitems , decision , questions 

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 


def split_transcript(transcript: str) -> list:
    # Split on natural boundaries where possible. The 200-character overlap
    # gives a finding near a chunk boundary a little context in both chunks.
    # RecursiveCharacterTextSplitter measures these sizes in characters here.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def get_llm():
    return ChatOpenAI(model = "gpt-4o-mini", openai_api_key = os.getenv("OPENAI_API_KEY"),temperature=0.2)



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def _extract_chunks(chunks: list, system_prompt: str, empty_result: str) -> str:
    if not chunks:
        return empty_result

    chain = build_chain(system_prompt)
    results = []

    # Each model call receives only one chunk, never the full transcript.
    for chunk in chunks:
        result = chain.invoke(chunk).strip()
        if result and result.casefold() != empty_result.casefold():
            results.append(result)

    return "\n\n".join(results) if results else empty_result


def extract_action_items(chunks: list) -> str:
    return _extract_chunks(
        chunks,
        "You are an expert transcript analyst. From this meeting, video, or podcast transcript, "
        "extract concrete action items, commitments, recommendations, or practical next steps "
        "that the speaker or a person/organization explicitly takes or recommends. "
        "Include advice clearly intended for the audience. For each provide:\n"
        "- Task description\n"
        "- Owner (person or organization, only if stated; otherwise 'Not specified')\n"
        "- Deadline (only if stated; otherwise 'Not specified')\n\n"
        "Do not turn general opinions, past events, or speculation into action items. "
        "Format as a numbered list. If none found say 'No action items found.'",
        "No action items found."
    )

def extract_key_decisions(chunks: list) -> str:
    return _extract_chunks(
        chunks,
        "You are an expert transcript analyst. From this meeting, video, or podcast transcript, "
        "extract key decisions and concrete changes that are described as chosen or implemented. "
        "Include reported company actions such as pricing changes, policy changes, experiments, "
        "launches, cancellations, and strategy shifts, even when the speaker does not call them "
        "'decisions'. Include who made each choice and its context when stated. "
        "Do not treat the speaker's opinion or speculation as a decision. "
        "Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        "No key decisions found."
    )


def extract_questions(chunks: list) -> str:
    return _extract_chunks(
        chunks,
        "From this meeting, video, or podcast transcript, extract questions that are explicitly "
        "asked and unresolved issues or uncertainties the speaker clearly raises. "
        "Do not invent questions or include topics that are already answered. "
        "Format as a numbered list. "
        "If none found say 'No open questions found.'",
        "No open questions found."
    )


def _consolidate_results(partial_results: str, category: str, empty_result: str) -> str:
    if not partial_results or partial_results.strip().casefold() == empty_result.casefold():
        return empty_result

    # Merge the chunk-level findings without sending the full transcript again.
    merge_chain = build_chain(
        f"You are combining partial {category} findings extracted from sections of one transcript. "
        "Treat the supplied findings as candidates. Preserve every distinct relevant candidate, "
        "merge only duplicates, and keep useful details and attribution. Do not discard a candidate "
        "just because it came from a video rather than a meeting. Do not invent information. "
        "Return one clean numbered list. If candidates are present, do not say that none were found. "
        f"If there are no findings, return exactly: {empty_result}"
    )

    current_text = partial_results

    # If the collected findings are too long for one call, merge them in batches,
    # then repeat on the shorter merged output for up to five rounds.
    for _ in range(5):
        result_batches = split_transcript(current_text)
        merged_results = []

        for result_batch in result_batches:
            merged = merge_chain.invoke(result_batch).strip()
            if merged and merged.casefold() != empty_result.casefold():
                merged_results.append(merged)

        if not merged_results:
            # The merge model must not erase findings from the chunk extractors.
            return current_text if current_text.strip() else empty_result
        if len(merged_results) == 1:
            return merged_results[0]

        current_text = "\n\n".join(merged_results)

    # Keep all partial findings if repeated batching did not reduce them enough.
    return current_text


def extract_meeting_items(transcript: str) -> dict:
    chunks = split_transcript(transcript)

    # First extract from every transcript chunk, then consolidate duplicates
    # and related findings without sending the full transcript to the model.
    action_items = extract_action_items(chunks)
    decisions = extract_key_decisions(chunks)
    questions = extract_questions(chunks)

    return {
        "action_items": _consolidate_results(
            action_items, "action items", "No action items found."
        ),
        "decisions": _consolidate_results(
            decisions, "key decisions", "No key decisions found."
        ),
        "questions": _consolidate_results(
            questions, "open questions", "No open questions found."
        ),
    }
