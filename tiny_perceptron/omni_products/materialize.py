"""從固定案例取出公開推論輸入；不把route/gold/family資訊送進產品。"""

from .choice import Candidate, Question
from .data import av_pair
from .routing import Request, b_sum_adapter


def public_question(value):
    candidates = tuple(Candidate(item["id"], item["description"]) for item in value["candidates"])
    return Question(value["state"], value["kind"], value["text"], candidates, gold=None).validate()


def public_request(value):
    if set(value) != {"text", "image_present", "audio_present", "video_present"}:
        raise ValueError("Auto request只能提供文字和三種附件presence")
    result = Request(**value)
    result.router_state()
    return result


def public_old_a(value):
    if not {"task", "messages"} <= set(value) or set(value) - {"task", "messages", "metadata", "tools"}:
        raise ValueError("舊A input只能有公開訊息、task、tools與媒體metadata")
    messages = []
    for message in value["messages"]:
        if set(message) - {"role", "content", "image", "audio", "roi", "image_layout"}:
            raise ValueError("公開訊息含未允許欄位")
        messages.append(dict(message))
    metadata = dict(value.get("metadata", {}))
    if set(metadata) - {"image", "audio", "roi", "image_layout", "modality_message_index"}:
        raise ValueError("未知公開媒體metadata")
    return {"task": value["task"], "messages": messages, "metadata": metadata, "tools": value.get("tools", False)}


def public_payload(value):
    if not value:
        return {}
    if set(value) == {"numbers"}:
        numbers = list(value["numbers"])
        b_sum_adapter(numbers)  # 驗有限文法，不計算答案。
        return {"numbers": numbers}
    if set(value) == {"questions"}:
        if set(value["questions"]) != {"choice", "info", "rubric", "choice_info"}:
            raise ValueError("C Auto案例須完整提供三型與對應Choice-info")
        return {"questions": {kind: public_question(question) for kind, question in value["questions"].items()}}
    if set(value) == {"old_A_input"}:
        return {"old_A_input": public_old_a(value["old_A_input"])}
    if set(value) == {"video_audio_definition"}:
        definition = value["video_audio_definition"]
        # 只按公開的生成／取樣定義還原媒體；產生器回傳的參考label丟棄。
        pair = av_pair(definition["pattern"], definition["variant"], definition["panel"])
        item, _ = pair[definition["which"]]
        return {"video_audio": item.validate()}
    raise ValueError("公開payload不得混入gold route/answer或其他隱藏欄位")


def auto_input(record):
    """兩條評測路線共用同一輸入；此介面完全不收gold route。"""
    return public_request(record["request"]), public_payload(record["payload"])
