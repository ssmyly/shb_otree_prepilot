from otree.api import Bot, Submission
from . import (
    C, Consent, Instructions, Practice, GetReady, Baseline, BaselineResult,
    Investment, TaskReady, Task, Results, FinalResults, Survey, SurveyFeedback, Exit,
)


class PlayerBot(Bot):
    def play_round(self):
        if self.round_number == 1:
            yield Submission(Consent, check_html=False)
            yield Submission(Instructions, check_html=False)
            yield Submission(Practice, {'practice_num_correct': 5,
                                        'practice_data_json': '[]'}, check_html=False)
            yield Submission(GetReady, check_html=False)
            yield Submission(Baseline, {'baseline_num_correct': 12,
                                        'baseline_data_json': '[]'}, check_html=False)
            yield Submission(BaselineResult, check_html=False)
        else:
            yield Submission(Investment, {'training_tokens': 2}, check_html=False)
            yield Submission(TaskReady, check_html=False)
            yield Submission(Task, {'num_correct': 11, 'task_data_json': '[]'},
                             check_html=False)
            yield Submission(Results, check_html=False)

        if self.round_number == C.NUM_ROUNDS:
            yield Submission(FinalResults, check_html=False)
            yield Submission(Survey, {
                'comp_check_tokens': 'They reduced the size of the counting grid',
                'comp_check_info': 'test',
                'survey_token_factors': 'test',
                'survey_change_over_rounds': 'test',
                'survey_gender': 'Prefer not to say',
                'survey_age': 30,
                'survey_education': "Bachelor's degree",
                'survey_employment': 'Full-time employed',
                'survey_income': 'Prefer not to say',
                'survey_job_search_freq': 'Sometimes (once a year)',
            }, check_html=False)
            yield Submission(SurveyFeedback, {
                'survey_experiment_thoughts': 'test',
                'survey_wording_strength': 'Neutral',
                'survey_wording_explain': 'test',
                'survey_user_friendly': 'test',
                'survey_problems': 'none',
                'survey_name': '',
            }, check_html=False)
            yield Submission(Exit, check_html=False)

            rounds = self.player.in_all_rounds()
            w0 = rounds[0].baseline_wage_offer
            work_net = sum(r.wage - r.training_cost for r in rounds[1:])
            expected = (C.BASE_PAY_POINTS + C.TOKEN_ENDOWMENT_POINTS + w0 + work_net)
            got = float(self.participant.payoff)
            assert abs(got - expected) < 1e-6, f"payoff mismatch: got {got}, expected {expected} (w0={w0}, work_net={work_net})"
            assert got >= C.BASE_PAY_POINTS - 1e-6, got
            print(f"[BOT OK] cond={self.participant.vars.get('condition')} payoff_pts={got:.2f} take_home=${got/C.POINTS_PER_USD:.2f}")
