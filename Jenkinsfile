def sendBuildStatusOverSlack() {
    // Sends a slack message based on various build parameters.
    def colorCode
    switch (currentBuild.result) {
        case 'SUCCESS':
            colorCode = 'good'
            break
        case 'FAILURE':
            colorCode = 'danger'
            break;
    }
    if (env.BRANCH == "main") {
        channelName = "simsci-ci-status"
    } else {
        channelName = "simsci-ci-status-test"
    }

    def job = Jenkins.getInstance().getItemByFullName(env.JOB_BASE_NAME, Job.class)
    def build = job.getBuildByNumber(env.BUILD_ID as int)
    def builder = build.getCause(Cause.UserIdCause).getUserId()
    // def builder = '???'
    // if (env.BUILD_USER_ID) {
    //     builder = env.BUILD_USER_ID
    //     builder -= ~/\@.*$/
    // }
    if (env.cron_user) {
         builder = 'Parameterized Cron'
    }
    def build_notes = ''
    if (env.build_notes) {
        build_notes += "Build notes: ${env.build_notes}"
    }
    def message = """
        Job: *${env.JOB_NAME}*
        ${build_notes}
        Build number: #${env.BUILD_NUMBER}
        Build status: *${currentBuild.result}*
        Author: @${builder}
        Build details: <${env.BUILD_URL}/console|See in web console>
    """.stripIndent()
    def slack_msg = [
        baseUrl:           "https://ihme.slack.com/services/hooks/jenkins-ci/",
        channel:           "${channelName}",
        tokenCredentialId: "slack",
        message:           message
    ]
    if (colorCode) { slack_msg.color = colorCode }
    return slackSend(slack_msg)
}


pipeline_name="vivarium_gates_nutrition_optimization"
conda_env_name="${pipeline_name}-${BUILD_NUMBER}"
conda_env_path="/tmp/${conda_env_name}"
// defaults for conda and pip are a local directory /svc-simsci for improved speed.
// In the past, we used /ihme/code/* on the NFS (which is slower)
shared_path="/svc-simsci"

pipeline {
  // This agent runs as svc-simsci on node simsci-slurm-sbuild-p01.
  // It has access to standard IHME filesystems and singularity
  agent { label "svc-simsci" }

  options {
    // Keep 100 old builds.
    buildDiscarder logRotator(numToKeepStr: "100")

    // Wait 60 seconds before starting the build.
    // If another commit enters the build queue in this time, the first build will be discarded.
    quietPeriod(60)

    // Fail immediately if any part of a parallel stage fails
    parallelsAlwaysFailFast()
  }

  parameters {
    string(
      name: "SLACK_TO",
      defaultValue: "simsci-ci-status",
      description: "The Slack channel to send messages to."
    )
    booleanParam(
      name: "DEBUG",
      defaultValue: false,
      description: "Used as needed for debugging purposes."
    )
  }

  environment {
    // Get the branch being built and strip everything but the text after the last "/"
    BRANCH = sh(script: "echo ${GIT_BRANCH} | rev | cut -d '/' -f1 | rev", returnStdout: true).trim()
    TIMESTAMP = sh(script: 'date', returnStdout: true)
    // Specify the path to the .condarc file via environment variable.
    // This file configures the shared conda package cache.
    CONDARC = "${shared_path}/miniconda3/.condarc"
    CONDA_BIN_PATH = "${shared_path}/miniconda3/bin"
    // Specify conda env by build number so that we don't have collisions if builds from
    // different branches happen concurrently.
    CONDA_ENV_NAME = "${conda_env_name}"
    CONDA_ENV_PATH = "${conda_env_path}"
    // Set the Pip cache.
    XDG_CACHE_HOME = "${shared_path}/pip-cache"
    // Jenkins commands run in separate processes, so need to activate the environment every
    // time we run pip, poetry, etc.
    ACTIVATE = "source ${CONDA_BIN_PATH}/activate ${CONDA_ENV_PATH} &> /dev/null"
  }

  stages {
    stage("Initialization") {
      steps {
        script {
          // Use the name of the branch in the build name
          currentBuild.displayName = "#${BUILD_NUMBER} ${GIT_BRANCH}"

          // Tell BitBucket that a build has started.
          notifyBitbucket()
        }
      }
    }

    stage("Debug Info") {
      steps {
        echo "Jenkins pipeline run timestamp: ${TIMESTAMP}"
        // Display parameters used.
        echo """Parameters:
        DEPLOY_OVERRIDE: ${params.DEPLOY_OVERRIDE}"""

        // Display environment variables from Jenkins.
        echo """Environment:
        ACTIVATE:       '${ACTIVATE}'
        BUILD_NUMBER:   '${BUILD_NUMBER}'
        BRANCH:         '${BRANCH}'
        CONDARC:        '${CONDARC}'
        CONDA_BIN_PATH: '${CONDA_BIN_PATH}'
        CONDA_ENV_NAME: '${CONDA_ENV_NAME}'
        CONDA_ENV_PATH: '${CONDA_ENV_PATH}'
        GIT_BRANCH:     '${GIT_BRANCH}'
        JOB_NAME:       '${JOB_NAME}'
        WORKSPACE:      '${WORKSPACE}'
        XDG_CACHE_HOME: '${XDG_CACHE_HOME}'"""
      }
    }

    stage("Build Environment") {
      environment {
        // Command for activating the base environment. Activating the base environment sets
        // the correct path to the conda binary which is used to create a new conda env.
        ACTIVATE_BASE = "source ${CONDA_BIN_PATH}/activate &> /dev/null"
      }
      steps {
        // The env should have been cleaned out after the last build, but delete it again
        // here just to be safe.
        sh "rm -rf ${CONDA_ENV_PATH}"
        sh "${ACTIVATE_BASE} && make build-env"
        // open permissions for test users to create file in workspace
        sh "chmod 777 ${WORKSPACE}"
      }
    }

    stage("Install Package") {
      steps {
        sh "${ACTIVATE} && make install"
      }
    }

    stage("Quality Checks") {
      parallel {
        stage("Format") {
          steps {
            sh "${ACTIVATE} && make format"
          }
        }

        stage("Lint") {
          steps {
            sh "${ACTIVATE} && make lint"
          }
        }
      }
    }

    stage("Test") {
      parallel {
        stage("Run Unit Tests") {
          steps {
            sh "${ACTIVATE} && make unit"
            publishHTML([
              allowMissing: true,
              alwaysLinkToLastBuild: false,
              keepAll: true,
              reportDir: "output/htmlcov_unit",
              reportFiles: "index.html",
              reportName: "Coverage Report - Unit Tests",
              reportTitles: ''
            ])
          }
        }

        stage("Run e2e Tests") {
          steps {
            sh "${ACTIVATE} && make e2e"
            publishHTML([
              allowMissing: true,
              alwaysLinkToLastBuild: false,
              keepAll: true,
              reportDir: "output/htmlcov_e2e",
              reportFiles: "index.html",
              reportName: "Coverage Report - e2e Tests",
              reportTitles: ''
            ])
          }
        }
      }
    }
  }

  post {
    always {
      sh "${ACTIVATE} && make clean"
      sh "rm -rf ${CONDA_ENV_PATH}"
      
      // Delete the workspace directory.
      deleteDir()
      // Send a message to Slack.
      sendBuildStatusOverSlack()
    }
    // failure {
    //   script {
    //     if (env.BRANCH == "main") {
    //       channelName = "simsci-ci-status"
    //     } else {
    //       channelName = "simsci-ci-status-test"
    //     }
    //   }
    //   // TODO: DM the developer instead of the slack channel
    //   echo "This build failed on ${GIT_BRANCH}. Sending a failure message to Slack."
    //   slackSend channel: "#${channelName}",
    //               message: ":x: JOB FAILURE: ${builder} triggered $JOB_NAME - $BUILD_ID\n\n${BUILD_URL}console\n\n<!channel>",
    //               teamDomain: "ihme",
    //               tokenCredentialId: "slack"
    // }
    success {
      script {
        if (params.DEBUG) {
          echo 'Debug is enabled. Sending a success message to Slack.'
          slackSend channel: "#${channelName}",
                    message: ":white_check_mark: (debugging) JOB SUCCESS: $JOB_NAME - $BUILD_ID\n\n${BUILD_URL}console",
                    teamDomain: "ihme",
                    tokenCredentialId: "slack"
        } else {
          echo 'Debug is not enabled. No success message will be sent to Slack.'
        }
      }
    }
  }
}