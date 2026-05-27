#!/bin/bash
# /opt/appinsights/jenkins-notify.sh
# Called as a post-build shell step in Jenkins jobs.
# Jenkins sets JOB_NAME, BUILD_RESULT, BUILD_NUMBER, BUILD_URL automatically.

# 1. App Insights — record every build result
python3 /opt/appinsights/ai-track.py \
  --role "jenkins" \
  --event "BuildCompleted" \
  --props "{\"job_name\":\"${JOB_NAME}\",\"result\":\"${BUILD_RESULT}\",\"build_number\":\"${BUILD_NUMBER}\",\"build_url\":\"${BUILD_URL}\"}"

# 2. Discord — notify only on failure
if [ "${BUILD_RESULT}" = "FAILURE" ]; then
  /opt/discord-bot/notify-discord.sh \
    "${JOB_NAME}" "FAILURE" "${BUILD_NUMBER}" "${BUILD_URL}"
fi
