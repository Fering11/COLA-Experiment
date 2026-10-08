from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable

from .client import LLMClient


OPTION_TO_STANCE = {"A": "against", "B": "favor", "C": "neutral"}
STANCES = ("favor", "against", "neutral")


@dataclass(frozen=True)
class PredictionResult:
    text: str
    target: str
    analyses: dict[str, str]
    debates: dict[str, str]
    judge_raw: str
    label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class COLAPipeline:
    """Three-stage local adaptation of the paper's single-round protocol."""

    def __init__(self, client: LLMClient, *, domain_role: str = "domain specialist"):
        self.client = client
        self.domain_role = domain_role
    
    def predict(
        self,
        *,
        text: str,
        target: str,
        id: str = "",
        printFunc: Callable | None = None,
    ) -> PredictionResult:
        if not text.strip():
            raise ValueError("text must not be empty")
        if not target.strip():
            raise ValueError("target must not be empty")
        
        if printFunc is None:
            printFunc = self._pass
        
        with ThreadPoolExecutor(max_workers=3) as pool:
            printFunc(f"{id}> analysing...")
            analysis_futures = {
                "linguistic": pool.submit(self._linguistic_analysis, text),
                "domain": pool.submit(self._domain_analysis, text, target),
                "social_media": pool.submit(self._social_media_analysis, text),
            }
            analyses = {name: future.result() for name, future in analysis_futures.items()}
            
            printFunc(f"{id}> debating...")
            debate_futures = {
                stance: pool.submit(
                    self._debate,
                    text=text,
                    target=target,
                    stance=stance,
                    analyses=analyses,
                )
                for stance in STANCES
            }
            debates = {stance: future.result() for stance, future in debate_futures.items()}
        judge_raw = self._judge(text=text, target=target, debates=debates)
        label = parse_judge_option(judge_raw)
        return PredictionResult(
            text=text,
            target=target,
            analyses=analyses,
            debates=debates,
            judge_raw=judge_raw,
            label=label,
        )
        
    def _pass(*args, **kwargs):
        pass

    def _linguistic_analysis(self, text: str) -> str:
        system = (
            "You are a linguist. Accurately and concisely explain the linguistic "
            "elements in the sentence and how these elements affect meaning, "
            "including grammatical structure, tense and inflection, virtual "
            "speech, rhetorical devices, lexical choices and so on. Do nothing else."
        )
        return self.client.complete(system=system, user=text)

    def _domain_analysis(self, text: str, target: str) -> str:
        system = (
            f"You are a {self.domain_role}. Accurately and concisely explain the "
            "key elements contained in the quote, such as characters, events, "
            "parties, religions, etc. Also explain their relationship with "
            f"{target} (if exist). Do nothing else."
        )
        return self.client.complete(system=system, user=text)

    def _social_media_analysis(self, text: str) -> str:
        system = (
            "You are a heavy social media user and are very familiar with the way "
            "of expression on the Internet. Analyze the following sentence, "
            "focusing on the content, emotional tone, implied meaning, and so on. "
            "Do nothing else."
        )
        return self.client.complete(system=system, user=text)

    def _debate(
        self,
        *,
        text: str,
        target: str,
        stance: str,
        analyses: dict[str, str],
    ) -> str:
        system = (
            "You are a stance debater. Defend only the stance assigned in the user "
            "message, using evidence from the provided analyses."
        )
        user = (
            f"Tweet: {text}\n"
            f"Linguistic analysis: {analyses['linguistic']}\n"
            f"The analysis of {self.domain_role}: {analyses['domain']}\n"
            "The analysis of a heavy social media user: "
            f"{analyses['social_media']}\n"
            f"You think the attitude behind the tweet is {stance} of {target}. "
            "Identify the top three pieces of evidence from the analyses that best "
            "support your opinion and argue for your opinion."
        )
        return self.client.complete(system=system, user=user)

    def _judge(self, *, text: str, target: str, debates: dict[str, str]) -> str:
        system = "You are the final stance judge."
        user = (
            f"Determine whether the sentence is in favor of or against {target}, "
            f"or is neutral. Sentence: {text}\n"
            "Judge this in relation to the following arguments:\n"
            f"Arguments that the attitude is in favor: {debates['favor']}\n"
            f"Arguments that the attitude is against: {debates['against']}\n"
            f"Arguments that the attitude is neutral: {debates['neutral']}\n"
            "Choose from: A: Against B: Favor C: Neutral\n"
            "Constraint: Answer with only the option above that is most accurate "
            "and nothing else."
        )
        return self.client.complete(system=system, user=user)


def parse_judge_option(raw: str) -> str:
    match = re.fullmatch(r"([ABC])(?:\s*:\s*(Against|Favor|Neutral))?[.!]?", raw.strip(), re.I)
    if match:
        label = OPTION_TO_STANCE[match.group(1).upper()]
        if match.group(2) and match.group(2).lower() != label:
            raise ValueError("Contradictory judge option and label")
        return label

    normalized = raw.strip().lower()
    for label in ("against", "favor", "neutral"):
        if normalized == label:
            return label
    raise ValueError(f"Could not parse judge output as A/B/C: {raw!r}")
