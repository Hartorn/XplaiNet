#!/bin/bash

# stop if error
set -e

read -p 'Release version: ' version
read -p 'You personal access token for Github: ' token
read -p 'Your username for pipy: ' pipyUser
read -p 'Your password for pipy: ' pipyPassword

echo ${version} | grep v && echo "Version should be x.y.z (for example, 1.1.1, 2.0.0, ...)" && exit -1

localDir=`readlink -f .`
releaseDir="${localDir}/release-${version}"
sudo rm -rf ${releaseDir}
mkdir ${releaseDir}
cd $releaseDir

echo "Cloning repo into XplaiNet"
git clone -q git@github.com:Hartorn/XplaiNet.git XplaiNet

cd XplaiNet
# Create release branch and push it
git checkout release/${version}

# Tagging proper version
echo "Tagging proper version"
git tag v${version}

# Build release
echo "Building latest build"
docker run --rm -v ${PWD}/xplainet:/work -w /work xplainet:latest poetry build

echo "Merging into develop and main"
git checkout main
git merge --no-ff origin/release/${version} -m "chore: release v${version} (merge)"
git checkout develop
git merge --no-ff origin/release/${version} -m "chore: release v${version} (merge)"

echo "Pushing branch"
git push origin develop
git push origin main
echo "Pushing tag"
git push origin --tags

echo "Making github release"
docker run -v ${PWD}:/work -w /work --entrypoint "" release-changelog:latest conventional-github-releaser -p angular --token ${token}

# Build release
echo "Building latest build"
docker run --rm -v ${PWD}/xplainet:/work -w /work xplainet:latest poetry build
# Build release
echo "Publishing latest build"
docker run --rm -v ${PWD}/xplainet:/work -w /work xplainet:latest poetry publish -u ${pipyUser} -p ${pipyPassword}

echo "Deleting release branch"
git checkout develop
git push origin :release/${version}

cd ${localDir}
sudo rm -rf ${releaseDir}