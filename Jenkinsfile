def githubUsernameToSlackName(github_author) {
  // Add team members as necessary
  def mapping = [
    "Jim Albright": "albrja",
    "Steve Bachmeier": "sbachmei",
    "Hussain Jafari": "hjafari",
    "Patrick Nast": "pnast",
    "Rajan Mudambi": "rmudambi",
  ]
  return mapping.get(github_author, "channel")
}

pipeline_name="vivarium_gates_nutrition_optimization"
conda_env_name="${pipeline_name}-${BRANCH_NAME}-${BUILD_NUMBER}"
conda_env_path="/tmp/${conda_env_name}"
// defaults for conda and pip are a local directory /svc-simsci for improved speed.
// In the past, we used /ihme/code/* on the NFS (which is slower)
shared_path="/svc-simsci"
// comma separated string list of branches to run periodic builds on

pipeline {
  // This agent runs as svc-simsci on simsci-jenkinsagent-ci-p01.
  // It has access to standard IHME filesystems and singularity
  agent { label "matrix-tasks" }

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
      // Generate a message to send to Slack.
      script {
        if (env.BRANCH == "main") {
          channelName = "simsci-ci-status"
        } else {
          channelName = "simsci-ci-status-test"
        }
        // Run git command to get the author of the last commit
        developerID = sh(
          script: "git log -1 --pretty=format:'%an'",
          returnStdout: true
        ).trim()
        slackID = githubUsernameToSlackName(developerID)
        slackMessage = """
          Job: *${env.JOB_NAME}*
          Build number: #${env.BUILD_NUMBER}
          Build status: *${currentBuild.result}*
          Author: @${slackID}
          Build details: <${env.BUILD_URL}/console|See in web console>
      """.stripIndent()
      }

      // Delete the workspace directory.
      deleteDir()
    }
    failure {
      echo "This build triggered by ${developerID} failed on ${GIT_BRANCH}. Sending a failure message to Slack."
      slackSend channel: "#${channelName}",
                  message: slackMessage,
                  teamDomain: "ihme",
                  tokenCredentialId: "slack"
    }
    success {
      script {
        if (params.DEBUG) {
          echo 'Debug is enabled. Sending a success message to Slack.'
          slackSend channel: "#${channelName}",
                    message: slackMessage,
                    teamDomain: "ihme",
                    tokenCredentialId: "slack"
        } else {
          echo 'Debug is not enabled. No success message will be sent to Slack.'
        }
      }
    }
  }
}
