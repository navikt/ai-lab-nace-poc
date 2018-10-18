#!/usr/bin/env groovy

import java.text.*

node {
    def app
    def yaml_path
    def version_tag

    def app_name = 'ai-lab-nace-poc'
    def namespace = 'ai-lab-nace-poc'
    def cluster = 'oera-q.local'

    def date = new Date()
    def datestring = new SimpleDateFormat("yyyy-MM-dd").format(date);

    try {
        stage('Clean workspace') {
            cleanWs()
        }

        stage('Checkout code') {
            checkout scm
            def git_commit_hash = sh (script: "git rev-parse --short HEAD", returnStdout: true)
            version_tag = "${datestring}-${git_commit_hash}"
        }

        stage('Copy certificate bundle and aws credentials') {
            sh "cp /etc/ssl/certs/ca-bundle.crt ca-bundle.crt"
            sh "cp /var/lib/jenkins/.awg/credentials aws-credentials"
        }

        stage('Build react app') {
            sh "chmod +x scripts/buildReactApp.sh"
            sh "./scripts/buildReactApp.sh"
        }

        stage('Build docker image') {
            app = docker.build("${app_name}", "-f Dockerfile.jenkins .")
        }

        stage('Upload nais.yaml to nexus server') {
            yaml_path = "https://repo.adeo.no/repository/raw/nais/${app_name}/${version_tag}/nais.yaml"
            sh "curl -s -S --upload-file nais.yaml ${yaml_path}"
        }

        stage('Push docker image') {
            docker.withRegistry('https://repo.adeo.no:5443', 'nexus-credentials') {
                app.push("${env.BUILD_ID}")
            }
        }

        stage('Deploy app to nais') {
            sh "curl --fail -k -d '{\"application\": \"${app_name}\", \"version\": \"${version_tag}\", \"skipFasit\": true, \"namespace\": \"${namespace}\", \"manifesturl\": \"${yaml_path}\"}' https://daemon.nais.${cluster}/deploy"
        }
    } catch(e) {
        echo "Build failed"
        throw e
    } finally {
        sh "docker rmi -f \$(docker images | grep 'repo.adeo.no:5443/${app_name}' | awk '{print \$3}') | true"
    }
}