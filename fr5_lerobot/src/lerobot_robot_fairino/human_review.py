"""Bind the user's explicit outcome decision to one immutable recorded episode.

Call record_review only after receiving the user's actual decision. A user
outcome is separate from technical data validity and does not itself permit
training, replay, or robot motion.
"""
import hashlib
import json
import time
from pathlib import Path


def resolve_review_evidence(episode, summary, *, original_episode=None):
    """Resolve episode-local evidence after a move without rewriting the summary.

    The capture report records the original journal path. An explicit original
    episode path supports older reports; neither permits basename-only guesses.
    """
    episode = Path(episode).resolve()
    roots = []
    journal = summary.get('continuous_capture', {}).get('metadata', {}).get('external_command_journal')
    if isinstance(journal, str) and Path(journal).is_absolute() and Path(journal).name == 'target-command-journal.json':
        roots.append(Path(journal).parent)
    if original_episode is not None:
        original = Path(original_episode)
        if not original.is_absolute():
            raise ValueError('Original episode path must be absolute')
        roots.append(original)
    resolved = []
    for item in summary.get('evidence', []):
        if not isinstance(item, str) or not item or '..' in Path(item).parts:
            raise ValueError('Invalid review evidence path')
        path = Path(item)
        if not path.is_absolute():
            candidate = (episode / path).resolve()
            if episode not in candidate.parents:
                raise ValueError('Review evidence escapes the episode')
        else:
            candidate = None
            for root in roots:
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                candidate = (episode / relative).resolve()
                if episode not in candidate.parents:
                    raise ValueError('Review evidence escapes the episode')
                break
            if candidate is None:
                candidate = path
        if not candidate.is_file():
            raise ValueError('Recorded review evidence is missing')
        resolved.append(candidate)
    return resolved


def _summary(episode, *, original_episode=None):
    episode=Path(episode)
    path=episode/'summary.json'
    raw=path.read_bytes()
    if (episode/'INCOMPLETE').exists():
        raise ValueError('Incomplete composite cannot receive a user review')
    summary=json.loads(raw)
    if summary.get('outcome') != 'pending_review':
        raise ValueError('Only a completed episode awaiting user review is eligible')
    if not summary.get('released') or summary.get('failed') or not summary.get('evidence'):
        raise ValueError('Incomplete episode cannot be labelled successful')
    if not summary.get('continuous_capture',{}).get('capture_complete'):
        raise ValueError('Continuous capture is incomplete')
    scope_raw=(episode/'scope.json').read_bytes()
    scope=json.loads(scope_raw)
    if (hashlib.sha256(scope_raw).hexdigest()!=summary.get('scope_sha256')
            or summary.get('completed_stages')!=scope.get('phase_order')
            or not summary.get('completed_stages')):
        raise ValueError('Scope identity or completed phase sequence is invalid')
    resolve_review_evidence(episode, summary, original_episode=original_episode)
    return summary,hashlib.sha256(raw).hexdigest()


def record_review(episode, *, verdict, user_statement, original_episode=None, input_method='chat_message'):
    episode=Path(episode)
    summary,digest=_summary(episode, original_episode=original_episode)
    if (verdict not in ('success','failure') or not isinstance(user_statement,str) or not user_statement.strip()
            or input_method not in ('chat_message', 'web_console_click')):
        raise ValueError('An explicit user statement and success or failure verdict are required')
    review={'schema':'fr5-user-review-v1','reviewer':'user','source':'direct_user_message',
            'input_method':input_method,
            'verdict':verdict,'user_statement':user_statement.strip(),
            'reviewed_at_unix':time.time(),'summary_sha256':digest,
            'scope_sha256':summary['scope_sha256'],
            'training_ready':False}
    with (episode/'user-review.json').open('x',encoding='utf-8') as file:
        json.dump(review,file,ensure_ascii=False,indent=2)
        file.write('\n')
    return review


def verify_success_review(episode, *, original_episode=None):
    episode=Path(episode)
    summary,digest=_summary(episode, original_episode=original_episode)
    raw=(episode/'user-review.json').read_bytes()
    review=json.loads(raw)
    if (review.get('schema')!='fr5-user-review-v1' or review.get('reviewer')!='user'
            or review.get('source')!='direct_user_message' or review.get('verdict')!='success'
            or not isinstance(review.get('user_statement'),str) or not review['user_statement'].strip()
            or review.get('summary_sha256')!=digest or review.get('scope_sha256')!=summary.get('scope_sha256')):
        raise ValueError('User success review is missing, rejected or refers to another episode')
    return review,hashlib.sha256(raw).hexdigest()
