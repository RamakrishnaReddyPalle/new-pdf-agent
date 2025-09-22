# packages/chat/splitter.py
from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from packages.chat.schemas import SplitPlan

_SPLIT_TMPL = PromptTemplate.from_template(
"""Split the user query into minimal atomic questions suitable for retrieval-augmented QA.
Preserve technical terms and references. Keep each sub-question independent.
{format_instructions}

USER_QUERY:
{query}
""")

def split_and_clean(llm, query: str) -> SplitPlan:
    parser = PydanticOutputParser(pydantic_object=SplitPlan)
    chain = _SPLIT_TMPL | llm | parser
    return chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
