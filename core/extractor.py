"""Extract actionable information from a transcript."""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


def split_transcript(transcript: str) -> list[str]:
    """Split a transcript into overlapping chunks for extraction."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def _build_chain(system_prompt: str):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda text: {"text": text})
        | prompt
        | llm
        | StrOutputParser()
    )


def _extract_chunks(chunks: list[str], system_prompt: str, empty_result: str) -> str:
    if not chunks:
        return empty_result

    chain = _build_chain(system_prompt)
    findings = []
    for chunk in chunks:
        result = chain.invoke(chunk).strip()
        if result and result.casefold() != empty_result.casefold():
            findings.append(result)

    return "\n\n".join(findings) if findings else empty_result


def _consolidate_results(partial_results: str, category: str, empty_result: str) -> str:
    if not partial_results or partial_results.strip().casefold() == empty_result.casefold():
        return empty_result

    merge_chain = _build_chain(
        f"You are combining partial {category} findings extracted from sections of one transcript. "
        "Treat the supplied findings as candidates. Preserve every distinct relevant candidate, "
        "merge only duplicates, and keep useful details and attribution. Do not invent information. "
        "Do not discard a candidate just because it came from a video rather than a meeting. "
        "Return one clean numbered list. If candidates are present, do not say that none were found. "
        f"If there are no findings, return exactly: {empty_result}"
    )

    current_text = partial_results
    for _ in range(5):
        result_batches = split_transcript(current_text)
        merged_results = []

        for result_batch in result_batches:
            merged = merge_chain.invoke(result_batch).strip()
            if merged and merged.casefold() != empty_result.casefold():
                merged_results.append(merged)

        if not merged_results:
            # Retain the extracted candidates if consolidation returns no usable text.
            return current_text if current_text.strip() else empty_result
        if len(merged_results) == 1:
            return merged_results[0]

        current_text = "\n\n".join(merged_results)

    return current_text


def _extract_transcript_category(
    transcript: str,
    system_prompt: str,
    category: str,
    empty_result: str,
) -> str:
    """Run chunk-level extraction and consolidate the findings for one category."""
    chunks = split_transcript(transcript)
    partial_results = _extract_chunks(chunks, system_prompt, empty_result)
    return _consolidate_results(partial_results, category, empty_result)


def extract_action_items(transcript: str) -> str:
    """Return transcript action items, commitments, and practical next steps."""
    return _extract_transcript_category(
        transcript,
        "You are an expert transcript analyst. From this meeting, video, or podcast transcript, "
        "extract concrete action items, commitments, recommendations, or practical next steps "
        "that the speaker or a person/organization explicitly takes or recommends. "
        "Include advice clearly intended for the audience. For each provide:\n"
        "- Task description\n"
        "- Owner (person or organization, only if stated; otherwise 'Not specified')\n"
        "- Deadline (only if stated; otherwise 'Not specified')\n\n"
        "Do not turn general opinions, past events, or speculation into action items. "
        "Format as a numbered list. If none found say 'No action items found.'",
        "action items",
        "No action items found.",
    )


def extract_key_decisions(transcript: str) -> str:
    """Return concrete decisions and changes described in the transcript."""
    return _extract_transcript_category(
        transcript,
        "You are an expert transcript analyst. From this meeting, video, or podcast transcript, "
        "extract key decisions and concrete changes that are described as chosen or implemented. "
        "Include reported company actions such as pricing changes, policy changes, experiments, "
        "launches, cancellations, and strategy shifts, even when the speaker does not call them "
        "'decisions'. Include who made each choice and its context when stated. "
        "Do not treat the speaker's opinion or speculation as a decision. "
        "Format as a numbered list. If none found say 'No key decisions found.'",
        "key decisions",
        "No key decisions found.",
    )


def extract_question(transcript: str) -> str:
    """Return explicit unanswered questions and unresolved issues in the transcript."""
    return _extract_transcript_category(
        transcript,
        "From this meeting, video, or podcast transcript, extract questions that are explicitly "
        "asked and unresolved issues or uncertainties the speaker clearly raises. "
        "Do not invent questions or include topics that are already answered. "
        "Format as a numbered list. If none found say 'No open questions found.'",
        "open questions",
        "No open questions found.",
    )
